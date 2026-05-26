
# LCP-LR: Lexical Complexity Prediction for Leveled Reading

This repository contains the official implementation and the LCP-LR benchmark dataset for the paper: **"Lexical Complexity Prediction for Leveled Reading."**

## Overview
LCP-LR aims to operationalize group-level personalization by identifying complex words relative to specific proficiency levels. This repository includes:
* **Benchmark Dataset:** 100,917 instance annotations across 1,401 leveled texts.
* **DPAL:** Dual-Prior Adversarial Learning framework incorporating word- and text-level difficulty priors with adversarial debiasing.
* **LLM Integration:** Fine-tuning scripts and API-based evaluation protocols for Large Language Models.

## Prerequisites
```bash
pip install -r requirements.txt

```

## Running Experiments

### 1. Baselines & DPAL

```bash
# XLNet-base baseline
python train.py --train_file data/train.jsonl --dev_file data/dev.jsonl --test_file data/test.jsonl

# DPAL (Proposed)
python train_adversarial.py --train_file data/train.jsonl --dev_file data/dev.jsonl --test_file data/test.jsonl

# Ablation studies
python ablation_unified.py --run_all --train_file data/train.jsonl

```

### 2. LLM Fine-tuning

```bash
cd llm_finetune
# Fine-tune LLMs (e.g., Qwen2.5-7B, Llama3-Chinese-8B)
python finetune_llm.py --model_type qwen --train_file ../data/train_llm.jsonl

# Evaluate LoRA adapter
python eval_llm.py --model_type qwen --adapter_path ./output/lora --test_file ../data/test_llm.jsonl

```

### 3. API Evaluation

```bash
cd llm_api
python eval_api.py --api_type openai --api_key YOUR_KEY --test_file ../data/test_llm.jsonl

```

## License

This project is licensed under the MIT License.

```
