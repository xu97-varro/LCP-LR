#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified ablation script for lexical complexity classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer, XLNetModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings

warnings.filterwarnings('ignore')


# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Studies for Lexical Complexity')
    parser.add_argument('--train_file', type=str, default='train.jsonl',
                        help='Training data file')
    parser.add_argument('--dev_file', type=str, default='dev.jsonl',
                        help='Validation data file')
    parser.add_argument('--test_file', type=str, default='test.jsonl',
                        help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./ablation_output',
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased',
                        help='Pretrained model name or path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max sequence length')
    parser.add_argument('--max_alpha', type=float, default=0.1,
                        help='Max adversarial coefficient')

    # Ablation mode - expanded
    parser.add_argument('--ablation_mode', type=str,
                        choices=[
                            'base',  # Main task only
                            'three_task',  # Main + text_aux + word_aux (no adversarial)
                            'adversarial',  # Full: both auxiliary + both adversarial
                            'word_adv_only',  # Word auxiliary + word adversarial only
                            'text_adv_only',  # Text auxiliary + text adversarial only
                            'no_adv',  # Both auxiliary tasks, no adversarial
                            'no_aux'  # Main task + both adversarial (aux only for adversarial)
                        ],
                        default='adversarial',
                        help='Ablation configuration')

    # Run all ablations
    parser.add_argument('--run_all', action='store_true',
                        help='Run all ablation modes sequentially')
    return parser.parse_args()


# ========== Gradient Reversal Layer ==========
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=0.1):
    return GradientReversalLayer.apply(x, alpha)


# ========== Level to Interval Mapping ==========
def get_text_level_interval(text_level):
    """Map text level (1-7) to 3 intervals: 1-3:0, 4-6:1, 7:2"""
    if text_level <= 3:
        return 0
    elif text_level <= 6:
        return 1
    else:
        return 2


def get_word_level_interval(word_level):
    """Map word level (1-7) to 3 intervals: 1-3:0, 4-6:1, 7:2"""
    if word_level <= 3:
        return 0
    elif word_level <= 6:
        return 1
    else:
        return 2


def get_alpha(epoch, max_alpha=0.1):
    """Dynamic adversarial coefficient: increase in first 3 epochs"""
    if epoch < 3:
        return max_alpha * (epoch + 1) / 3
    else:
        return max_alpha


# ========== Data Loading ==========
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ========== Model Builder ==========
class XLNetAblationModel(nn.Module):
    """Unified model for all ablation configurations"""

    def __init__(self, model_name, ablation_mode):
        super().__init__()
        self.encoder = XLNetModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.ablation_mode = ablation_mode

        # Main task: lexical complexity (4 classes)
        self.difficulty_classifier = nn.Linear(hidden_size, 4)

        # Auxiliary tasks (7 classes each)
        self.use_text_aux = ablation_mode in ['three_task', 'adversarial', 'text_adv_only', 'no_adv']
        self.use_word_aux = ablation_mode in ['three_task', 'adversarial', 'word_adv_only', 'no_adv']

        if self.use_text_aux:
            self.text_level_classifier = nn.Linear(hidden_size, 7)
        if self.use_word_aux:
            self.word_level_classifier = nn.Linear(hidden_size, 7)

        # Adversarial heads (3 classes each)
        self.use_adv_text = ablation_mode in ['adversarial', 'text_adv_only']
        self.use_adv_word = ablation_mode in ['adversarial', 'word_adv_only']

        if self.use_adv_text:
            self.adversarial_text_classifier = nn.Linear(hidden_size, 3)
        if self.use_adv_word:
            self.adversarial_word_classifier = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask, alpha=0.1):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, -1, :]
        pooled = self.dropout(pooled)

        # Main task
        difficulty_logits = self.difficulty_classifier(pooled)

        # Auxiliary tasks
        text_level_logits = self.text_level_classifier(pooled) if self.use_text_aux else None
        word_level_logits = self.word_level_classifier(pooled) if self.use_word_aux else None

        # Adversarial tasks
        if self.use_adv_text or self.use_adv_word:
            pooled_adv = grad_reverse(pooled, alpha)
            adv_text_logits = self.adversarial_text_classifier(pooled_adv) if self.use_adv_text else None
            adv_word_logits = self.adversarial_word_classifier(pooled_adv) if self.use_adv_word else None
        else:
            adv_text_logits = None
            adv_word_logits = None

        return difficulty_logits, text_level_logits, word_level_logits, adv_text_logits, adv_word_logits


# ========== Dataset Class ==========
class AblationDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, ablation_mode, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ablation_mode = ablation_mode
        self.difficulty2id = {'简单': 0, '可读': 1, '较难': 2, '困难': 3}

        self.use_text_aux = ablation_mode in ['three_task', 'adversarial', 'text_adv_only', 'no_adv']
        self.use_word_aux = ablation_mode in ['three_task', 'adversarial', 'word_adv_only', 'no_adv']
        self.use_adv_text = ablation_mode in ['adversarial', 'text_adv_only']
        self.use_adv_word = ablation_mode in ['adversarial', 'word_adv_only']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"Word: {item['target_word']} Context: {item['context']}"

        encoding = self.tokenizer(
            input_text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'difficulty_label': torch.tensor(self.difficulty2id[item['label']], dtype=torch.long),
        }

        if self.use_text_aux or self.use_adv_text:
            result['text_level_label'] = torch.tensor(item['text_level'] - 1, dtype=torch.long)
            result['text_interval'] = torch.tensor(get_text_level_interval(item['text_level']), dtype=torch.long)

        if self.use_word_aux or self.use_adv_word:
            word_level = item.get('word_level', 1)
            result['word_level_label'] = torch.tensor(word_level - 1, dtype=torch.long)
            result['word_interval'] = torch.tensor(get_word_level_interval(word_level), dtype=torch.long)

        return result


def collate_fn(batch):
    result = {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'difficulty_labels': torch.stack([b['difficulty_label'] for b in batch]),
    }
    if 'text_level_label' in batch[0]:
        result['text_level_labels'] = torch.stack([b['text_level_label'] for b in batch])
        result['text_intervals'] = torch.stack([b['text_interval'] for b in batch])
    if 'word_level_label' in batch[0]:
        result['word_level_labels'] = torch.stack([b['word_level_label'] for b in batch])
        result['word_intervals'] = torch.stack([b['word_interval'] for b in batch])
    return result


# ========== Training and Evaluation ==========
def train_and_evaluate(args, mode):
    """Train and evaluate model for a specific ablation mode"""

    print(f"\n{'=' * 60}")
    print(f"Running ablation: {mode}")
    print(f"{'=' * 60}")

    # Load data
    train_data = load_jsonl(args.train_file)
    dev_data = load_jsonl(args.dev_file)
    test_data = load_jsonl(args.test_file)

    print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    # Initialize model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained(args.model_name)
    model = XLNetAblationModel(args.model_name, mode)

    # Create datasets
    train_dataset = AblationDataset(train_data, tokenizer, mode, args.max_length)
    dev_dataset = AblationDataset(dev_data, tokenizer, mode, args.max_length)
    test_dataset = AblationDataset(test_data, tokenizer, mode, args.max_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss functions
    class_weights = torch.tensor([0.3, 3.5, 20.0, 7.0], dtype=torch.float).to(device)
    main_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    aux_loss_fn = nn.CrossEntropyLoss()
    adv_loss_fn = nn.CrossEntropyLoss()

    # Loss weights based on mode
    if mode == 'three_task':
        lambda_text, lambda_word = 0.3, 0.2
        lambda_adv_text, lambda_adv_word = 0, 0
    elif mode == 'adversarial':
        lambda_text, lambda_word = 0.1, 0.1
        lambda_adv_text, lambda_adv_word = 0.05, 0.05
    elif mode == 'word_adv_only':
        lambda_text, lambda_word = 0, 0.1
        lambda_adv_text, lambda_adv_word = 0, 0.05
    elif mode == 'text_adv_only':
        lambda_text, lambda_word = 0.1, 0
        lambda_adv_text, lambda_adv_word = 0.05, 0
    elif mode == 'no_adv':
        lambda_text, lambda_word = 0.1, 0.1
        lambda_adv_text, lambda_adv_word = 0, 0
    else:  # base or no_aux
        lambda_text, lambda_word = 0, 0
        lambda_adv_text, lambda_adv_word = 0, 0

    print(
        f"Loss weights - text: {lambda_text}, word: {lambda_word}, adv_t: {lambda_adv_text}, adv_w: {lambda_adv_word}")

    # Training loop
    best_f1 = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        alpha = get_alpha(epoch, args.max_alpha) if (model.use_adv_text or model.use_adv_word) else 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = model(inputs['input_ids'], inputs['attention_mask'], alpha)
            difficulty_logits, text_logits, word_logits, adv_text_logits, adv_word_logits = outputs

            loss = main_loss_fn(difficulty_logits, inputs['difficulty_labels'])

            if text_logits is not None and 'text_level_labels' in inputs:
                loss += lambda_text * aux_loss_fn(text_logits, inputs['text_level_labels'])
                if adv_text_logits is not None and 'text_intervals' in inputs:
                    loss += lambda_adv_text * adv_loss_fn(adv_text_logits, inputs['text_intervals'])

            if word_logits is not None and 'word_level_labels' in inputs:
                loss += lambda_word * aux_loss_fn(word_logits, inputs['word_level_labels'])
                if adv_word_logits is not None and 'word_intervals' in inputs:
                    loss += lambda_adv_word * adv_loss_fn(adv_word_logits, inputs['word_intervals'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                difficulty_logits, _, _, _, _ = model(inputs['input_ids'], inputs['attention_mask'], alpha=0)
                preds = torch.argmax(difficulty_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(inputs['difficulty_labels'].cpu().numpy())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Dev F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_{mode}.pt'))

    # Test evaluation
    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_{mode}.pt'), map_location=device))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            difficulty_logits, _, _, _, _ = model(inputs['input_ids'], inputs['attention_mask'], alpha=0)
            preds = torch.argmax(difficulty_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(inputs['difficulty_labels'].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Results - Acc: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    per_class = {label: round(per_class_f1[i], 4) for i, label in enumerate(['简单', '可读', '较难', '困难'])}

    return {
        'mode': mode,
        'test_accuracy': float(accuracy),
        'test_macro_f1': float(macro_f1),
        'test_weighted_f1': float(weighted_f1),
        'per_class_f1': per_class,
        'best_dev_f1': float(best_f1)
    }


# ========== Main ==========
def main():
    args = parse_args()

    if args.run_all:
        # Run all ablation modes
        modes = [
            'base',  # Main task only
            'three_task',  # Three-task (text + word auxiliary)
            'no_adv',  # Auxiliary tasks only, no adversarial
            'word_adv_only',  # Word auxiliary + word adversarial
            'text_adv_only',  # Text auxiliary + text adversarial
            'adversarial'  # Full model (both auxiliary + both adversarial)
        ]

        all_results = {}
        for mode in modes:
            try:
                result = train_and_evaluate(args, mode)
                all_results[mode] = result
                print(f"\n✓ Completed: {mode} -> Macro F1: {result['test_macro_f1']:.4f}")
            except Exception as e:
                print(f"\n✗ Failed: {mode} - {e}")
                all_results[mode] = {'error': str(e)}

        # Save all results
        results_path = os.path.join(args.output_dir, 'all_ablation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print("All Ablation Results Summary")
        print(f"{'=' * 60}")
        for mode, res in all_results.items():
            if 'error' not in res:
                print(f"{mode:15s}: Macro F1 = {res['test_macro_f1']:.4f}")
            else:
                print(f"{mode:15s}: ERROR")
        print(f"\nResults saved to {results_path}")

    else:
        # Run single mode
        result = train_and_evaluate(args, args.ablation_mode)

        # Save result
        results_path = os.path.join(args.output_dir, f'{args.ablation_mode}_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()