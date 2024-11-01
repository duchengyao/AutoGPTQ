from logging import getLogger

import torch
import torch.nn as nn


logger = getLogger(__name__)

nf4_map = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

nf4_tensor={}
def get_nf4_tensor(x):
    keys=nf4_tensor.keys()
    if x.device not in keys:
        nf4_tensor[x.device]=torch.tensor(nf4_map, device=x.device, dtype=torch.float32)
    return nf4_tensor[x.device]


def quantize_nf4(x, scale, zero, maxq):
    DEBUG = False

    if DEBUG:
        if maxq != 15:
            print("[Warning]!!!!![quantize_nf4] maxq!=15,", maxq)

        if x.dim() != 2:
            print("[Warning]!!!!![quantize_nf4] x.dim()!=2,", x.dim)

        if x.shape[-1] != 1:
            print("[Warning]!!!!![quantize_nf4] x.shape[-1]!=1,", x.shape)

    nf4_tensor = get_nf4_tensor(x)
    expanded_nf4_tensor = nf4_tensor.unsqueeze(1)

    # 量化
    norm_0to16 = ((x / scale) + zero)
    norm_n1to1 = norm_0to16 / 8 - 1
    q = nf4_tensor[torch.abs(expanded_nf4_tensor - norm_n1to1.squeeze(1).unsqueeze(0)).argmin(dim=0)].unsqueeze(
        1)
    # 反量化
    dq = scale * ((q + 1) * 8 - zero)

    if DEBUG:
        diff = dq - quantize(x, scale, zero, maxq)
        if any(max(abs(diff * scale)) > 0.3):
            print("[Warning]!!!!max(diff):", max(diff))

    return dq


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        print("=====Warning: maxq is negative=====")
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        nf4=False,
        trits=False,
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.nf4 = nf4
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.nf4:
            return quantize_nf4(x, self.scale, self.zero, self.maxq)
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer"]

"""
unit test
"""
if __name__ == '__main__':
    x = torch.tensor([[0], [7.8796e-2]])
    scale = torch.tensor([[1], [0.0438]])
    zero = torch.tensor([[0], [8]])
    maxq = torch.tensor(15)

    print(quantize(x, scale, zero, maxq))
    print(quantize_nf4(x, scale, zero, maxq))
