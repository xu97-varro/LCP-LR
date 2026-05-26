#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning LLM for lexical complexity classification using LoRA.
Supports Qwen2.5 and Llama3 models.
"""

import json
import torch
import os
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import argparse


# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune LLM for Lexical Complexity')

    # Data paths
    parser.add_argument('--train_file', type=str, default='train_llm.jsonl',
                        help='Training data file')
    parser.add_argument('--dev_file', type=str, default='dev_llm.jsonl',
                        help='Validation data file')
    parser.add_argument('--test_file', type=str, default='test_llm.jsonl',
                        help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./finetune_output',
                        help='Output directory')

    # Model selection
    parser.add_argument('--model_type', type=str, choices=['qwen', 'llama3'], default='qwen',
                        help='Model type: qwen or llama3')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Pretrained model name or path (overrides model_type)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Per device training batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=384,
                        help='Maximum sequence length')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_balanced_data', action='store_true',
                        help='Use balanced training data')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ========== System Prompt ==========
SYSTEM_PROMPT = """You are an expert in International Chinese Education.

Text levels range from 1 to 7 (1 = simplest, 7 = most difficult), indicating the overall language complexity of the context and the target learner level.

Difficulty definitions:
- Simple: The word is far below the learner's current level, posing no comprehension barrier
- Readable: The word is within the learner's expected vocabulary range, understandable without assistance
- Difficult: The word is slightly above the learner's current level, requiring contextual inference or minimal explanation
- Very Difficult: The word is far above the learner's current level, posing significant comprehension barrier even with context"""


def format_chat_sample(sample, include_answer=True):
    """Format sample for chat model"""
    user_content = f"""Target word: {sample['target_word']}
Context: {sample['context']}
Text level: {sample['text_level']}

Please judge the difficulty of this word for learners at level {sample['text_level']}. Output only one word: Simple / Readable / Difficult / Very Difficult"""

    if include_answer:
        # Map labels to English
        label_map = {'简单': 'Simple', '可读': 'Readable', '较难': 'Difficult', '困难': 'Very Difficult'}
        answer = label_map.get(sample['label'], sample['label'])

        conversations = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    else:
        conversations = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
    return {"conversations": conversations}


# ========== Model-specific Chat Templates ==========
def apply_qwen_chat_template(conversations):
    """Convert conversations to Qwen chat format"""
    text = ""
    for msg in conversations:
        if msg["role"] == "system":
            text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    return text


def apply_llama3_chat_template(conversations):
    """Convert conversations to Llama3 chat format"""
    text = ""
    for msg in conversations:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text


# ========== Main ==========
def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Set model name
    if args.model_name is None:
        if args.model_type == 'qwen':
            model_name = 'Qwen/Qwen2.5-7B-Instruct'
        else:
            model_name = 'FlagAlpha/Llama3-Chinese-8B-Instruct'
    else:
        model_name = args.model_name

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    lora_output_dir = os.path.join(args.output_dir, 'lora')

    # ========== 1. Load Data ==========
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    train_data = load_jsonl(args.train_file)
    dev_data = load_jsonl(args.dev_file)
    test_data = load_jsonl(args.test_file)

    print(f"Train: {len(train_data)} samples")
    print(f"Dev: {len(dev_data)} samples")
    print(f"Test: {len(test_data)} samples")

    # Label distribution
    label_counts = {'Simple': 0, 'Readable': 0, 'Difficult': 0, 'Very Difficult': 0}
    label_map_cn = {'简单': 'Simple', '可读': 'Readable', '较难': 'Difficult', '困难': 'Very Difficult'}
    for sample in train_data:
        label_en = label_map_cn[sample['label']]
        label_counts[label_en] += 1

    print(f"\nTrain label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count / len(train_data) * 100:.1f}%)")

    # ========== 2. Format Data ==========
    print("\nFormatting data...")

    train_formatted = [format_chat_sample(s, include_answer=True) for s in train_data]
    dev_formatted = [format_chat_sample(s, include_answer=True) for s in dev_data]
    test_formatted = [format_chat_sample(s, include_answer=False) for s in test_data]

    print("\nTraining data example:")
    print(json.dumps(train_formatted[0], ensure_ascii=False, indent=2))

    # ========== 3. Load Model with Quantization ==========
    print("\nLoading model...")
    print("=" * 60)

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=(args.model_type == 'qwen'),
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=(args.model_type == 'qwen'))

    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ========== 4. LoRA Configuration ==========
    print("\nConfiguring LoRA...")

    # Target modules for different model architectures
    if args.model_type == 'qwen':
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:  # llama3
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # ========== 5. Data Preprocessing ==========
    print("\nPreprocessing data...")

    # Choose chat template
    if args.model_type == 'qwen':
        apply_template = apply_qwen_chat_template
    else:
        apply_template = apply_llama3_chat_template

    def preprocess_function(examples):
        texts = []
        for conv in examples["conversations"]:
            text = apply_template(conv)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Mask padding tokens in labels
        pad_token_id = tokenizer.pad_token_id
        labels = tokenized["labels"]
        labels = [[-100 if token == pad_token_id else token for token in label] for label in labels]
        tokenized["labels"] = labels

        return tokenized

    train_dataset = Dataset.from_list(train_formatted)
    dev_dataset = Dataset.from_list(dev_formatted)

    remove_cols = [col for col in train_dataset.column_names if col != "conversations"]
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # ========== 6. Training Arguments ==========
    effective_batch_size = args.batch_size * args.gradient_accumulation

    training_args = TrainingArguments(
        output_dir=lora_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        logging_dir=os.path.join(args.output_dir, "logs"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    print(f"\nTraining configuration:")
    print(f"  Model: {model_name}")
    print(f"  Model type: {args.model_type}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Batch size: {args.batch_size} × {args.gradient_accumulation} = {effective_batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max length: {args.max_length}")
    print(f"  LoRA r/alpha: {args.lora_r}/{args.lora_alpha}")

    # ========== 7. Train ==========
    print("\nStarting training...")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # ========== 8. Save Model ==========
    print(f"\nSaving LoRA model to {lora_output_dir}")
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()