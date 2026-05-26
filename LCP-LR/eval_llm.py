#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified evaluation script for fine-tuned LLM models.
Supports both Qwen2.5 and Llama3 models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import os
import argparse
import warnings

warnings.filterwarnings('ignore')


# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned LLM models')

    # Model paths
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path to base model (if not using --model_type)')
    parser.add_argument('--model_type', type=str, choices=['qwen', 'llama3'], default='qwen',
                        help='Model type: qwen or llama3')
    parser.add_argument('--adapter_path', type=str, required=True,
                        help='Path to LoRA adapter weights')

    # Data
    parser.add_argument('--test_file', type=str, default='test_llm.jsonl',
                        help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for results')

    # Generation
    parser.add_argument('--max_new_tokens', type=int, default=20,
                        help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Generation temperature')

    return parser.parse_args()


# ========== System Prompt ==========
SYSTEM_PROMPT = """You are an expert in International Chinese Education.

Text levels range from 1 to 7 (1 = simplest, 7 = most difficult), indicating the overall language complexity of the context and the target learner level.

Difficulty definitions:
- Simple: The word is far below the learner's current level, posing no comprehension barrier
- Readable: The word is within the learner's expected vocabulary range, understandable without assistance
- Difficult: The word is slightly above the learner's current level, requiring contextual inference or minimal explanation
- Very Difficult: The word is far above the learner's current level, posing significant comprehension barrier even with context"""

# Label mapping (Chinese to ID)
LABEL_MAP = {'简单': 0, '可读': 1, '较难': 2, '困难': 3}
ID_TO_LABEL = {0: '简单', 1: '可读', 2: '较难', 3: '困难'}
TARGET_NAMES = ['简单', '可读', '较难', '困难']


def load_model(base_model_path, adapter_path, model_type):
    """Load base model and LoRA adapter"""
    print(f"Loading base model from: {base_model_path}")
    print(f"Loading adapter from: {adapter_path}")
    print(f"Model type: {model_type}")

    # Load tokenizer
    trust_remote_code = (model_type == 'qwen')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)

    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return tokenizer, model


def predict_qwen(sample, tokenizer, model, max_new_tokens=20, temperature=0.1):
    """Predict using Qwen chat template"""
    user_content = f"""Target word: {sample['target_word']}
Context: {sample['context']}
Text level: {sample['text_level']}

Please judge the difficulty of this word for learners at level {sample['text_level']}. Output only one word: Simple / Readable / Difficult / Very Difficult"""

    conversation = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(conversation, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # Map English output to Chinese labels
    if 'Simple' in response:
        return '简单'
    elif 'Readable' in response:
        return '可读'
    elif 'Difficult' in response and 'Very' not in response:
        return '较难'
    elif 'Very Difficult' in response:
        return '困难'
    # Fallback: try to match Chinese directly
    for label in ['简单', '可读', '较难', '困难']:
        if label in response:
            return label
    return None


def predict_llama3(sample, tokenizer, model, max_new_tokens=20, temperature=0.1):
    """Predict using Llama3 chat template"""
    user_content = f"""Target word: {sample['target_word']}
Context: {sample['context']}
Text level: {sample['text_level']}

Please judge the difficulty of this word for learners at level {sample['text_level']}. Output only one word: Simple / Readable / Difficult / Very Difficult"""

    conversation = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(conversation, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # Map English output to Chinese labels
    if 'Simple' in response:
        return '简单'
    elif 'Readable' in response:
        return '可读'
    elif 'Difficult' in response and 'Very' not in response:
        return '较难'
    elif 'Very Difficult' in response:
        return '困难'
    # Fallback: try to match Chinese directly
    for label in ['简单', '可读', '较难', '困难']:
        if label in response:
            return label
    return None


def load_test_data(test_file):
    """Load test data from JSONL file"""
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate_model(tokenizer, model, test_data, model_type, max_new_tokens=20, temperature=0.1):
    """Evaluate model on test data"""
    predict_fn = predict_qwen if model_type == 'qwen' else predict_llama3

    y_true = []
    y_pred = []
    invalid_count = 0

    for sample in tqdm(test_data, desc="Evaluating"):
        pred = predict_fn(sample, tokenizer, model, max_new_tokens, temperature)

        y_true.append(LABEL_MAP[sample['label']])

        if pred is not None and pred in LABEL_MAP:
            y_pred.append(LABEL_MAP[pred])
        else:
            y_pred.append(-1)
            invalid_count += 1

    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(y_pred) if p != -1]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    return {
        'y_true': y_true_valid,
        'y_pred': y_pred_valid,
        'total_samples': len(test_data),
        'valid_samples': len(y_true_valid),
        'invalid_samples': invalid_count
    }


def print_results(results, model_name):
    """Print detailed evaluation results"""
    y_true = results['y_true']
    y_pred = results['y_pred']

    print(f"\n{'=' * 70}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'=' * 70}")
    print(f"Total samples: {results['total_samples']}")
    print(
        f"Valid predictions: {results['valid_samples']} ({results['valid_samples'] / results['total_samples'] * 100:.1f}%)")
    print(
        f"Invalid predictions: {results['invalid_samples']} ({results['invalid_samples'] / results['total_samples'] * 100:.1f}%)")

    if results['valid_samples'] == 0:
        print("No valid predictions!")
        return None

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy   : {acc:.4f}")
    print(f"   Macro F1   : {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")

    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"{'':>12} {'简单':>8} {'可读':>8} {'较难':>8} {'困难':>8}")
    for i, name in enumerate(TARGET_NAMES):
        print(f"{name:>12} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8} {cm[i][3]:>8}")

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = []
    for i in range(4):
        if sum(cm[i]) > 0:
            per_class_recall.append(cm[i][i] / sum(cm[i]))
        else:
            per_class_recall.append(0.0)

    print(f"\n📈 Per-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    for i, name in enumerate(TARGET_NAMES):
        precision = cm[i][i] / sum(cm[:, i]) if sum(cm[:, i]) > 0 else 0
        print(f"{name:<12} {precision:<12.4f} {per_class_recall[i]:<12.4f} {per_class_f1[i]:<12.4f}")

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_f1': {name: float(per_class_f1[i]) for i, name in enumerate(TARGET_NAMES)},
        'per_class_recall': {name: float(per_class_recall[i]) for i, name in enumerate(TARGET_NAMES)},
        'confusion_matrix': cm.tolist(),
        'valid_samples': results['valid_samples'],
        'total_samples': results['total_samples']
    }


def main():
    args = parse_args()

    # Set default base model path based on model type
    if args.base_model is None:
        if args.model_type == 'qwen':
            base_model = 'Qwen/Qwen2.5-7B-Instruct'
        else:
            base_model = 'FlagAlpha/Llama3-Chinese-8B-Instruct'
    else:
        base_model = args.base_model

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print("=" * 60)
    print("Loading test data...")
    print("=" * 60)
    test_data = load_test_data(args.test_file)
    print(f"Test samples: {len(test_data)}")

    # Load model
    print("\nLoading model...")
    tokenizer, model = load_model(base_model, args.adapter_path, args.model_type)

    # Evaluate
    results = evaluate_model(
        tokenizer, model, test_data, args.model_type,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # Print results
    model_name = os.path.basename(args.adapter_path)
    metrics = print_results(results, model_name)

    # Save results
    if metrics:
        output_file = os.path.join(args.output_dir, f'{model_name}_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")


# ========== Compare Multiple Models ==========
def compare_models(args_list):
    """Compare multiple models (for batch evaluation)"""
    all_results = {}

    for model_args in args_list:
        print(f"\n{'#' * 70}")
        print(f"Evaluating: {model_args['name']}")
        print(f"{'#' * 70}")

        # Load test data
        test_data = load_test_data(model_args['test_file'])

        # Load model
        tokenizer, model = load_model(
            model_args['base_model'],
            model_args['adapter_path'],
            model_args['model_type']
        )

        # Evaluate
        results = evaluate_model(
            tokenizer, model, test_data, model_args['model_type'],
            max_new_tokens=model_args.get('max_new_tokens', 20),
            temperature=model_args.get('temperature', 0.1)
        )

        # Calculate metrics
        if results['valid_samples'] > 0:
            y_true = results['y_true']
            y_pred = results['y_pred']
            all_results[model_args['name']] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro'),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
                'valid_samples': results['valid_samples'],
                'total_samples': results['total_samples']
            }

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("📊 Model Comparison Summary")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Valid Rate':<12}")
    print("-" * 56)
    for name, res in all_results.items():
        valid_rate = res['valid_samples'] / res['total_samples']
        print(f"{name:<20} {res['accuracy']:<12.4f} {res['macro_f1']:<12.4f} {valid_rate:<12.2%}")

    return all_results


if __name__ == "__main__":
    # Single model evaluation
    # python eval_llm.py --model_type qwen --adapter_path ./output/lora --test_file test.jsonl

    # Or uncomment below for comparing multiple models:
    """
    models_to_evaluate = [
        {
            'name': 'Qwen-20k-original',
            'model_type': 'qwen',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
            'adapter_path': './qwen_20k_output/lora',
            'test_file': 'test_llm.jsonl'
        },
        {
            'name': 'Qwen-20k-balanced',
            'model_type': 'qwen',
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
            'adapter_path': './qwen_balanced_output/lora',
            'test_file': 'test_llm.jsonl'
        },
        {
            'name': 'Llama3-20k-original',
            'model_type': 'llama3',
            'base_model': 'FlagAlpha/Llama3-Chinese-8B-Instruct',
            'adapter_path': './llama3_20k_output/lora',
            'test_file': 'test_llm.jsonl'
        },
        {
            'name': 'Llama3-20k-balanced',
            'model_type': 'llama3',
            'base_model': 'FlagAlpha/Llama3-Chinese-8B-Instruct',
            'adapter_path': './llama3_balanced_output/lora',
            'test_file': 'test_llm.jsonl'
        }
    ]
    compare_models(models_to_evaluate)
    """

    main()