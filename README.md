# LCP-LR: Lexical Complexity Prediction for Leveled Reading

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202026-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and the LCP-LR benchmark dataset for the EMNLP 2026 paper:

**"Beyond Over-Simplification and Over-Personalization: Lexical Complexity Prediction for Leveled Reading"**

> Lexical Complexity Prediction (LCP) in real-world scenarios, particularly in language education, is often limited by the trade-off between overly simplified general models and unscalable personalized ones. To address this, we introduce **LCP-LR**, a task that operationalizes group-level personalization by identifying complex words relative to specific proficiency levels. We construct the first large-scale benchmark dataset (**100,917** instance annotations across **1,401** leveled texts) and propose **DPAL** (Dual-Prior Adversarial Learning), which incorporates word- and text-level difficulty priors with adversarial debiasing. Our framework significantly outperforms PLMs and LLMs baselines, achieving **77.81%** Macro F1.
> 

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{xu2026lexical,
  title={Beyond Over-Simplification and Over-Personalization: Lexical Complexity Prediction for Leveled Reading},
  author={Huidan Xu and Ying Liu},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}
```

## Overview

LCP-LR operationalizes group-level personalization by identifying complex words relative to specific proficiency levels. This repository includes:

- **Benchmark Dataset:** 100,917 instance annotations across 1,401 leveled texts.
- **DPAL:** Dual-Prior Adversarial Learning framework incorporating word- and text-level difficulty priors with adversarial debiasing.
- **LLM Integration:** Fine-tuning scripts and API-based evaluation protocols for Large Language Models.

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the National Social Science Fund of China under Grant No. 23BYY198, and the Tsinghua University Graduate Education Reform Project 2026 under Grant No. 202604J023.

## Contact

For questions or issues, please open an [issue](https://github.com/xu97-varro/LCP-LR/issues) or contact the authors directly.

```
