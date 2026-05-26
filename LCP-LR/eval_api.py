#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API-based evaluation for lexical complexity classification.
Supports GPT-4o-mini and Qwen3-Max.
"""

import json
import requests
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import argparse
from tqdm import tqdm


# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LLM API for Lexical Complexity')

    # API configuration
    parser.add_argument('--api_type', type=str, choices=['openai', 'dashscope'], required=True,
                        help='API type: openai (GPT-4o-mini) or dashscope (Qwen3-Max)')
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for the LLM service')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (auto-selected if not specified)')

    # Data
    parser.add_argument('--test_file', type=str, default='test_llm.jsonl',
                        help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./api_results',
                        help='Output directory for results')

    # Prompt strategy
    parser.add_argument('--strategy', type=str,
                        choices=['zero_shot', 'few_shot', 'few_shot_cot', 'all'],
                        default='few_shot',
                        help='Prompting strategy')

    # Testing
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (for quick testing)')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between API calls (seconds)')

    return parser.parse_args()


# ========== Prompt Templates ==========

# Zero-shot prompt
ZERO_SHOT_PROMPT = """You are an expert in International Chinese Education.

Text levels range from 1 to 7 (1 = simplest, 7 = most difficult), indicating the overall language complexity of the context and the target learner level.

Target word: {target_word}
Context: {context}
Text level: {text_level}

Please judge the difficulty of this word for learners at level {text_level}.

Difficulty definitions:
- Simple: The word is far below the learner's current level, posing no comprehension barrier
- Readable: The word is within the learner's expected vocabulary range, understandable without assistance
- Difficult: The word is slightly above the learner's current level, requiring contextual inference or minimal explanation
- Very Difficult: The word is far above the learner's current level, posing significant comprehension barrier even with context

Output only one word: Simple / Readable / Difficult / Very Difficult"""

# Few-shot prompt
FEW_SHOT_PROMPT = """You are an expert in International Chinese Education.

Text levels range from 1 to 7 (1 = simplest, 7 = most difficult), indicating the overall language complexity of the context and the target learner level.

Here are some examples:

Example 1:
Target word: scenic_spot
Context: China has a vast territory with beautiful mountains and rivers. Thousands of scenic spots are distributed across this vast land.
Text level: 7
Difficulty: Simple

Example 2:
Target word: competition
Context: The North Wind and the Sun held a competition to see who could make a traveler remove his coat.
Text level: 3
Difficulty: Readable

Example 3:
Target word: scenic_spot
Context: This is a big city in China. It's very old and famous worldwide. Many people come here to visit a scenic spot called the Terracotta Warriors.
Text level: 4
Difficulty: Difficult

Example 4:
Target word: envy
Context: He went to China last year and studied in Beijing for a year. Now he speaks Chinese very well. I really envy him.
Text level: 3
Difficulty: Very Difficult

Difficulty definitions:
- Simple: The word is far below the learner's current level, posing no comprehension barrier
- Readable: The word is within the learner's expected vocabulary range, understandable without assistance
- Difficult: The word is slightly above the learner's current level, requiring contextual inference or minimal explanation
- Very Difficult: The word is far above the learner's current level, posing significant comprehension barrier even with context

Now judge the following:

Target word: {target_word}
Context: {context}
Text level: {text_level}

Output only one word: Simple / Readable / Difficult / Very Difficult"""

# Few-shot CoT prompt
COT_FEW_SHOT_PROMPT = """You are an expert in International Chinese Education.

Text levels range from 1 to 7 (1 = simplest, 7 = most difficult), indicating the overall language complexity of the context and the target learner level.

Difficulty definitions:
- Simple: Word level < text level, no comprehension barrier
- Readable: Word level ≈ text level, understandable without assistance
- Difficult: Word level > text level, requires contextual inference
- Very Difficult: Word level >> text level, significant barrier even with context

Read the following examples carefully, including the reasoning process.

【Example 1】
Target word: scenic_spot
Context: China has a vast territory with beautiful mountains and rivers. Thousands of scenic spots are distributed across this vast land.
Text level: 7
Reasoning:
1. Word complexity assessment: "scenic_spot" is an intermediate-level word.
2. Text difficulty assessment: Level 7 targets advanced learners.
3. Match analysis: Intermediate word in advanced text, word difficulty is below text level.
4. Judgment: For advanced learners, this word poses no comprehension barrier.
Answer: Simple

【Example 2】
Target word: competition
Context: The North Wind and the Sun held a competition to see who could make a traveler remove his coat.
Text level: 3
Reasoning:
1. Word complexity assessment: "competition" is a basic-intermediate word.
2. Text difficulty assessment: Level 3 targets beginner learners.
3. Match analysis: Word complexity roughly matches text level.
4. Judgment: For beginner learners, this word is within expected range.
Answer: Readable

【Example 3】
Target word: scenic_spot
Context: This is a big city in China. It's very old and famous worldwide. Many people come here to visit a scenic spot called the Terracotta Warriors.
Text level: 4
Reasoning:
1. Word complexity assessment: "scenic_spot" is an intermediate word.
2. Text difficulty assessment: Level 4 targets beginner-intermediate learners.
3. Match analysis: Intermediate word in beginner-intermediate text, slightly above level.
4. Judgment: Requires contextual inference or minimal explanation.
Answer: Difficult

【Example 4】
Target word: envy
Context: He went to China last year and studied in Beijing for a year. Now he speaks Chinese very well. I really envy him.
Text level: 3
Reasoning:
1. Word complexity assessment: "envy" is an advanced word.
2. Text difficulty assessment: Level 3 targets beginner learners.
3. Match analysis: Advanced word in beginner text, far above learner level.
4. Judgment: Poses significant comprehension barrier even with context.
Answer: Very Difficult

---

Now judge the following:

Target word: {target_word}
Context: {context}
Text level: {text_level}

Please reason step by step:
1. Word complexity assessment
2. Text difficulty assessment
3. Match analysis
4. Judgment

Output format:
Reasoning: [your reasoning process]
Answer: Simple / Readable / Difficult / Very Difficult"""

# ========== Label Mapping ==========
EN_TO_CN = {
    'Simple': '简单',
    'Readable': '可读',
    'Difficult': '较难',
    'Very Difficult': '困难'
}

LABEL_TO_ID = {'简单': 0, '可读': 1, '较难': 2, '困难': 3}
TARGET_NAMES = ['简单', '可读', '较难', '困难']


def get_prompt_template(strategy):
    """Get prompt template for given strategy"""
    templates = {
        'zero_shot': ZERO_SHOT_PROMPT,
        'few_shot': FEW_SHOT_PROMPT,
        'few_shot_cot': COT_FEW_SHOT_PROMPT
    }
    return templates.get(strategy)


def extract_answer(response_text, strategy):
    """Extract answer from model response"""
    if not response_text:
        return None

    # For CoT, extract from "Answer:" line
    if strategy == 'few_shot_cot' and 'Answer:' in response_text:
        for line in response_text.split('\n'):
            if 'Answer:' in line:
                for eng_label in ['Simple', 'Readable', 'Difficult', 'Very Difficult']:
                    if eng_label in line:
                        return EN_TO_CN.get(eng_label)

    # Direct matching for English labels
    for eng_label, cn_label in EN_TO_CN.items():
        if eng_label in response_text:
            return cn_label

    # Fallback: try to match Chinese directly
    for cn_label in ['简单', '可读', '较难', '困难']:
        if cn_label in response_text:
            return cn_label

    return None


# ========== API Callers ==========

def call_openai_api(prompt, api_key, model_name, max_retries=3):
    """Call OpenAI GPT-4o-mini API"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name or "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)

            if resp.status_code == 200:
                result = resp.json()
                response = result["choices"][0]["message"]["content"].strip()
                return response
            else:
                print(f"OpenAI API error (attempt {attempt + 1}): {resp.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except Exception as e:
            print(f"OpenAI request error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def call_dashscope_api(prompt, api_key, model_name, max_retries=3):
    """Call Alibaba DashScope Qwen3-Max API"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name or "qwen-max",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)

            if resp.status_code == 200:
                result = resp.json()
                response = result["choices"][0]["message"]["content"].strip()
                return response
            else:
                print(f"DashScope API error (attempt {attempt + 1}): {resp.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except Exception as e:
            print(f"DashScope request error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def load_data(file_path):
    """Load test data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate_strategy(strategy, test_data, api_type, api_key, model_name,
                      output_dir, delay=0.05, max_samples=None):
    """Evaluate a single prompting strategy"""

    strategy_name = f"{api_type}_{strategy}"

    print(f"\n{'=' * 70}")
    print(f"API: {api_type.upper()}")
    print(f"Model: {model_name or ('gpt-4o-mini' if api_type == 'openai' else 'qwen-max')}")
    print(f"Strategy: {strategy}")
    print(f"{'=' * 70}")

    # Limit samples if specified
    if max_samples and max_samples < len(test_data):
        test_data = test_data[:max_samples]
        print(f"Using first {max_samples} samples for testing")

    print(f"Total samples: {len(test_data)}")

    prompt_template = get_prompt_template(strategy)
    if prompt_template is None:
        print(f"Unknown strategy: {strategy}")
        return None

    # Select API caller
    if api_type == 'openai':
        call_api = call_openai_api
    else:
        call_api = call_dashscope_api

    y_true = []
    y_pred = []
    results = []

    for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {strategy}")):
        # Format prompt
        prompt = prompt_template.format(
            target_word=sample['target_word'],
            context=sample['context'],
            text_level=sample['text_level']
        )

        # Call API
        response = call_api(prompt, api_key, model_name)

        # Extract answer
        pred_label = extract_answer(response, strategy) if response else None

        true_label = sample['label']
        y_true.append(LABEL_TO_ID[true_label])

        if pred_label and pred_label in LABEL_TO_ID:
            y_pred.append(LABEL_TO_ID[pred_label])
            results.append({
                'target_word': sample['target_word'],
                'context': sample['context'][:100] + '...' if len(sample['context']) > 100 else sample['context'],
                'text_level': sample['text_level'],
                'true_label': true_label,
                'pred_label': pred_label,
                'correct': pred_label == true_label,
                'raw_response': response[:200] if response else None
            })
        else:
            y_pred.append(-1)
            results.append({
                'target_word': sample['target_word'],
                'true_label': true_label,
                'pred_label': None,
                'correct': False,
                'error': 'Failed to extract answer'
            })

        # Rate limiting delay
        time.sleep(delay)

    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(y_pred) if p != -1]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    results_valid = [results[i] for i in valid_indices]

    print(
        f"\nValid predictions: {len(y_true_valid)}/{len(test_data)} ({len(y_true_valid) / len(test_data) * 100:.1f}%)")

    if len(y_true_valid) == 0:
        print("No valid predictions!")
        return None

    # Calculate metrics
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    macro_f1 = f1_score(y_true_valid, y_pred_valid, average='macro')
    weighted_f1 = f1_score(y_true_valid, y_pred_valid, average='weighted')

    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy   : {accuracy:.4f}")
    print(f"   Macro F1   : {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")

    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true_valid, y_pred_valid, target_names=TARGET_NAMES, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    print(f"\n📊 Confusion Matrix:")
    print(f"{'':>12} {'简单':>8} {'可读':>8} {'较难':>8} {'困难':>8}")
    for i, name in enumerate(TARGET_NAMES):
        print(f"{name:>12} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8} {cm[i][3]:>8}")

    # Per-class F1
    per_class_f1 = f1_score(y_true_valid, y_pred_valid, average=None, zero_division=0)
    print(f"\n📈 Per-Class F1:")
    for i, name in enumerate(TARGET_NAMES):
        print(f"   {name}: {per_class_f1[i]:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{api_type}_{strategy}_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'api_type': api_type,
            'model': model_name or ('gpt-4o-mini' if api_type == 'openai' else 'qwen-max'),
            'strategy': strategy,
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'per_class_f1': {name: float(per_class_f1[i]) for i, name in enumerate(TARGET_NAMES)},
            'confusion_matrix': cm.tolist(),
            'valid_samples': len(y_true_valid),
            'total_samples': len(test_data),
            'results': results_valid
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print examples
    correct_results = [r for r in results_valid if r.get('correct')]
    wrong_results = [r for r in results_valid if not r.get('correct')]
    print(f"\n✅ Correct: {len(correct_results)} | ❌ Wrong: {len(wrong_results)}")

    if correct_results:
        print("\nSample correct predictions:")
        for r in correct_results[:3]:
            print(f"   {r['target_word']}: true={r['true_label']}, pred={r['pred_label']}")

    if wrong_results:
        print("\nSample wrong predictions:")
        for r in wrong_results[:3]:
            print(f"   {r['target_word']}: true={r['true_label']}, pred={r['pred_label']}")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1.tolist() if hasattr(per_class_f1, 'tolist') else per_class_f1
    }


def main():
    args = parse_args()

    # Set default model name if not specified
    if args.model_name is None:
        if args.api_type == 'openai':
            args.model_name = 'gpt-4o-mini'
        else:
            args.model_name = 'qwen-max'

    print("=" * 60)
    print("LLM API Evaluation for Lexical Complexity")
    print("=" * 60)
    print(f"API Type: {args.api_type.upper()}")
    print(f"Model: {args.model_name}")
    print(f"Strategy: {args.strategy}")
    print(f"Test file: {args.test_file}")

    # Load test data
    print("\nLoading test data...")
    test_data = load_data(args.test_file)
    print(f"Loaded {len(test_data)} samples")

    # Run evaluation
    if args.strategy == 'all':
        strategies = ['zero_shot', 'few_shot', 'few_shot_cot']
        all_results = {}

        for strategy in strategies:
            result = evaluate_strategy(
                strategy, test_data, args.api_type, args.api_key, args.model_name,
                args.output_dir, args.delay, args.max_samples
            )
            if result:
                all_results[strategy] = result

        # Print comparison
        print(f"\n{'=' * 70}")
        print(f"📊 {args.api_type.upper()} - Strategy Comparison Summary")
        print(f"{'=' * 70}")
        print(f"{'Strategy':<15} {'Accuracy':<12} {'Macro F1':<12}")
        print("-" * 40)
        for strategy, res in all_results.items():
            print(f"{strategy:<15} {res['accuracy']:<12.4f} {res['macro_f1']:<12.4f}")

        # Save comparison
        comparison_file = os.path.join(args.output_dir, f'{args.api_type}_strategy_comparison.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")

    else:
        evaluate_strategy(
            args.strategy, test_data, args.api_type, args.api_key, args.model_name,
            args.output_dir, args.delay, args.max_samples
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()