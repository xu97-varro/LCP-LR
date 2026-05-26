# Lexical Complexity Prediction for Leveled Reading

This repository contains code for LCP-LR, including:
- XLNet-base with multi-task adversarial learning
- Fine-tuning LLMs (Qwen2.5-7B, Llama3-Chinese-8B)
- API-based evaluation (GPT-4o-mini, Qwen-Max)

## Usage

```bash
pip install -r requirements.txt

# XLNet baseline
python train.py --train_file data/train.jsonl --dev_file data/dev.jsonl --test_file data/test.jsonl

# DPAL (proposed)
python train_adversarial.py --train_file data/train.jsonl --dev_file data/dev.jsonl --test_file data/test.jsonl

# Ablation studies
python ablation_unified.py --run_all --train_file data/train.jsonl

# LLM fine-tuning
cd llm_finetune
python finetune_llm.py --model_type qwen --train_file ../data/train_llm.jsonl
python eval_llm.py --model_type qwen --adapter_path ./output/lora --test_file ../data/test_llm.jsonl

# API evaluation
cd llm_api
python eval_api.py --api_type openai --api_key YOUR_KEY --test_file ../data/test_llm.jsonl
