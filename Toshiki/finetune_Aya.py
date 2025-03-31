#this python code is written based on the following codes:
#https://github.com/DoJunggeun/contrastivemix
#https://huggingface.co/CohereForAI/aya-23-8B/blob/main/Aya_23_notebook.ipynb

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from datasets import Dataset
import random
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import re
import wandb
import logging
from functools import partial
from transformers import (
    HfArgumentParser,
    set_seed,
)
from konlpy.tag import Komoran
import MeCab
from mecab import MeCab
from huggingface_hub import login
from arguments import ModelArguments, DataArguments
from trl import SFTConfig

logger = logging.getLogger(__name__)

def main():
    huggingface_token = "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"
    login(huggingface_token)
    wandb_key = "c2079b970735a0f41c87351ad4844489ec12df1d"
    wandb.login(key=wandb_key)
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args= parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)


    #MODEL_NAME = "CohereForAI/aya-23-8b"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, use_fast=True,
    )

    # Load Model
    quantization_config = None
    if model_args.quantize_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    attn_implementation = None
    if model_args.use_flash_attention:
        attn_implementation="flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )

    '''
    ds_jaa = load_dataset("CohereForAI/aya_collection_language_split", "japanese", cache_dir=model_args.cache_dir)
    ds_koa = load_dataset("CohereForAI/aya_collection_language_split", "korean", cache_dir=model_args.cache_dir)

    ds_ja = Dataset.from_dict(ds_jaa["train"][0:2])
    ds_ko = Dataset.from_dict(ds_koa["train"][0:2])

    dataset = concatenate_datasets([ds_ja, ds_ko]).shuffle(model_args.seed)
    '''
    dataset = load_dataset("json", data_files="./data/finetuning_dataset.json").shuffle(model_args.seed)
    formatted_data = dataset.map(formatting_prompts_func, batched=True)
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048
        )

        tokens["labels"] = tokens["input_ids"].copy()
        tokens["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in tokens["labels"]
        ]

        return tokens

    tokenized_data = formatted_data.map(tokenize_function, batched=True)
    dataset = tokenized_data["train"]
    #dataset = Dataset.from_dict(dataset[0:2])
    '''
    komoran = Komoran()
    mecab_wakati = MeCab.Tagger("-Owakati")
    src2tgt_ja2ko = get_dict("ja-ko")
    src2tgt_ko2ja = get_dict("ko-ja")
    
    def codemix_train(example):
        if example["language"] == "kor":
            if (
                not data_args.codemix_in_runtime
                and data_args.codemix_ratio > 0
                and random.random() < data_args.codemix_sentence_ratio
            ):
                example["inputs"] = get_codemixed_ko2ja(
                    komoran,
                    src2tgt_ko2ja,
                    example["inputs"],
                    model_args.train_max_seq_length,
                    data_args.codemix_ratio,
                )
                example["targets"] = get_codemixed_ko2ja(
                    komoran,
                    src2tgt_ko2ja,
                    example["targets"],
                    model_args.train_max_seq_length,
                    data_args.codemix_ratio,
                )
                return example

            return example

        else:
            if (
                not data_args.codemix_in_runtime
                and data_args.codemix_ratio > 0
                and random.random() < data_args.codemix_sentence_ratio
            ):
                example["inputs"] = get_codemixed_ja2ko(
                    mecab_wakati,
                    src2tgt_ja2ko,
                    example['inputs'],
                    model_args.train_max_seq_length,
                    data_args.codemix_ratio,
                )
                example["targets"] = get_codemixed_ja2ko(
                    mecab_wakati,
                    src2tgt_ja2ko,
                    example["targets"],
                    model_args.train_max_seq_length,
                    data_args.codemix_ratio,
                )
                return example

            return example

    dataset = dataset["train"].map(
        codemix_train,
        batched=False,
        num_proc=data_args.dataset_proc_num,
        desc="Performing codemixing",
    )'
    '''

    # Training Arguments
    training_arguments = SFTConfig(
        output_dir="results",
        num_train_epochs=model_args.num_epochs,
        per_device_train_batch_size=model_args.train_batch_size,
        gradient_accumulation_steps=model_args.grad_acc_steps,
        gradient_checkpointing=model_args.use_grad_checkpointing,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=1e-3,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        seed=model_args.seed,
        max_seq_length=model_args.train_max_seq_length,
    )

    peft_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        r=model_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_arguments,
        #formatting_func=formatting_prompts_func
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {model_args.num_epochs}")

    trainer.train()

    # Save the model to disk
    trainer.model.save_pretrained(save_directory=f'{model_args.new_model_name}')
    model2 = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )
    model2 = PeftModel.from_pretrained(model2, f'./{model_args.new_model_name}')
    model2.push_to_hub(f"tona3738/{model_args.new_model_name}")
    model.config.use_cache = True
    model.eval()

'''
def get_dict(codemix_set):
    dict_path = os.path.join(os.path.dirname(__file__), "dict", f"{codemix_set}.txt")
    lines = open(dict_path, "r", encoding="utf-8").readlines()
    src2tgt = {}
    for line in lines:
        line = line.strip()
        try:
            src, tgt = line.split("\t")
        except:
            src, tgt = line.split(" ")

        if src not in src2tgt:
            src2tgt[src] = [tgt]
        else:
            src2tgt[src].append(tgt)
    return src2tgt


def get_codemixed_ja2ko(
    basic_tokenizer, src2tgt, input, max_length, codemix_ratio=0
):
    if codemix_ratio == 0:
        return input
    
    words = basic_tokenizer.parse(input).split()
    codemixed = ""
    for w in words:
        if (random.random() < codemix_ratio) and w in src2tgt:
            trans_w = src2tgt[w][random.randint(0, len(src2tgt[w]) - 1)]
            codemixed += trans_w
        else:
            codemixed += w
    return codemixed

def get_codemixed_ko2ja(
    basic_tokenizer, src2tgt, input, max_length, codemix_ratio=0
):
    if codemix_ratio == 0:
        return input
    
    words = basic_tokenizer.morphs(input)
    codemixed = ""
    for w in words:
        if (random.random() < codemix_ratio) and w in src2tgt:
            trans_w = src2tgt[w][random.randint(0, len(src2tgt[w]) - 1)]
            codemixed += trans_w
        else:
            codemixed += w
    return codemixed
'''
'''
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['inputs'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{example['inputs'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['targets'][i]}"
        output_texts.append(text)
    return output_texts
'''
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['inputs'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{example['inputs'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['targets'][i]}"
        output_texts.append(text)
    return {"text": output_texts}

if __name__ == "__main__":
    main()