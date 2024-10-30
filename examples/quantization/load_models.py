import torch
import transformers

from transformers import AutoTokenizer, TextGenerationPipeline

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from auto_gptq.utils import Perplexity

import numpy as np
import torch.nn as nn

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit-128g"


# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


def find_layers(module, layers=None, name=""):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_module_by_name_prefix(model, module_name: str):
    for name, module in model.named_modules():
        if name.startswith(module_name):
            return module


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

traindataset, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)

quantized_int4_nosafetensors = AutoGPTQForCausalLM.from_quantized("quantized_int4_nosafetensors", device="cuda:0",use_safetensors=False)
quantized_int4_safetensors = AutoGPTQForCausalLM.from_quantized("quantized_int4_safetensors", device="cuda:1")
quantized_int4_unpack_nosafetensors = AutoGPTQForCausalLM.from_quantized("quantized_int4_unpack_nosafetensors", device="cuda:2",use_safetensors=False)
quantized_int4_unpack_safetensors = AutoGPTQForCausalLM.from_quantized("quantized_int4_unpack_safetensors", device="cuda:3")

pass
