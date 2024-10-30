import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, BitsAndBytesConfig

# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from auto_gptq.utils import Perplexity

pretrained_model_dir = "facebook/opt-125m"


def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)



    model1 = AutoModelForCausalLM.from_pretrained(
        "gptq_int4_safetensors",
    )
    ppl1 = Perplexity(
        model1,
        tokenizer
    )

    ppl1.calculate_perplexity()

    model2 = AutoModelForCausalLM.from_pretrained(
        "bnb-nf4",
    )
    ppl2 = Perplexity(
        model2,
        tokenizer
    )

    ppl2.calculate_perplexity()
    return

    model1 = AutoModelForCausalLM.from_pretrained(
        pretrained_model_dir,
        # quantization_config=BitsAndBytesConfig(
            # load_in_4bit=False,
            # bnb_4bit_compute_dtype=torch.float16
        # ),
        device_map="auto",
    )

    model1.save_pretrained("bnb-float")


    model2 = AutoModelForCausalLM.from_pretrained(
        pretrained_model_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ),
        device_map="auto",
    )

    model2.save_pretrained("bnb-nf4")

    return

    model3 = AutoModelForCausalLM.from_pretrained(
        pretrained_model_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ),
        device_map="auto",
    )

    ppl = Perplexity(
        model1,
        tokenizer
    )

    ppl.calculate_perplexity()

    ppl = Perplexity(
        model2,
        tokenizer
    )

    ppl.calculate_perplexity()

    ppl = Perplexity(
        model3,
        tokenizer
    )

    ppl.calculate_perplexity()


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
