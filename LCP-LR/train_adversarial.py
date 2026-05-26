#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-task learning with adversarial training for lexical complexity classification.
Submitted to [Conference Name - Anonymized].
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


# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Lexical Complexity Classification with Adversarial Training')
    parser.add_argument('--train_file', type=str, default='train.jsonl',
                        help='Training data file (with level annotations)')
    parser.add_argument('--dev_file', type=str, default='dev.jsonl',
                        help='Validation data file (with level annotations)')
    parser.add_argument('--test_file', type=str, default='test.jsonl',
                        help='Test data file (with level annotations)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for model and results')
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased',
                        help='Pretrained model name or path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--lambda_text', type=float, default=0.1,
                        help='Weight for text level auxiliary task')
    parser.add_argument('--lambda_word', type=float, default=0.1,
                        help='Weight for word level auxiliary task')
    parser.add_argument('--lambda_adv_text', type=float, default=0.05,
                        help='Weight for text adversarial task')
    parser.add_argument('--lambda_adv_word', type=float, default=0.05,
                        help='Weight for word adversarial task')
    parser.add_argument('--max_alpha', type=float, default=0.1,
                        help='Maximum adversarial coefficient')
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
    """
    Map text level (1-7) to 3 intervals:
    1-3: beginner (0)
    4-6: intermediate (1)
    7:   advanced (2)
    """
    if text_level <= 3:
        return 0
    elif text_level <= 6:
        return 1
    else:
        return 2


def get_word_level_interval(word_level):
    """
    Map word level (1-7) to 3 intervals:
    1-3: beginner (0)
    4-6: intermediate (1)
    7:   advanced (2)
    """
    if word_level <= 3:
        return 0
    elif word_level <= 6:
        return 1
    else:
        return 2


# ========== Dynamic Adversarial Coefficient ==========
def get_alpha(epoch, total_epochs=5, max_alpha=0.1):
    """
    Dynamic adversarial coefficient:
    Linearly increase from 0 to max_alpha in first 3 epochs,
    then keep max_alpha for remaining epochs.

    Args:
        epoch: current epoch (0-indexed)
        total_epochs: total number of epochs
        max_alpha: maximum adversarial coefficient
    """
    if epoch < 3:
        # Linear increase: epoch0: max_alpha/3, epoch1: 2*max_alpha/3, epoch2: max_alpha
        return max_alpha * (epoch + 1) / 3
    else:
        # Keep max for remaining epochs
        return max_alpha


# ========== Label Mappings ==========
DIFFICULTY2ID = {'简单': 0, '可读': 1, '较难': 2, '困难': 3}
ID2DIFFICULTY = {0: '简单', 1: '可读', 2: '较难', 3: '困难'}

# Class weights based on inverse frequency in training set
CLASS_WEIGHTS = torch.tensor([0.3, 3.5, 20.0, 7.0], dtype=torch.float)


# ========== 1. Data Loading ==========
def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ========== 2. Multi-Task Model with Adversarial Heads ==========
class XLNetMultiTaskWithAdversarial(nn.Module):
    """XLNet-based multi-task model with adversarial training"""

    def __init__(self, model_name):
        super().__init__()
        self.encoder = XLNetModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Main task: lexical complexity classification (4 classes)
        self.difficulty_classifier = nn.Linear(hidden_size, 4)

        # Auxiliary task 1: passage level classification (7 classes)
        self.text_level_classifier = nn.Linear(hidden_size, 7)

        # Auxiliary task 2: word level classification (7 classes)
        self.word_level_classifier = nn.Linear(hidden_size, 7)

        # Adversarial head 1: predict passage level interval (3 classes)
        self.adversarial_text_classifier = nn.Linear(hidden_size, 3)

        # Adversarial head 2: predict word level interval (3 classes)
        self.adversarial_word_classifier = nn.Linear(hidden_size, 3)

    def forward(self, input_ids, attention_mask, alpha=0.1):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use last token representation (similar to [CLS] for XLNet)
        pooled = outputs.last_hidden_state[:, -1, :]
        pooled = self.dropout(pooled)

        # Main and auxiliary tasks (normal forward)
        difficulty_logits = self.difficulty_classifier(pooled)
        text_level_logits = self.text_level_classifier(pooled)
        word_level_logits = self.word_level_classifier(pooled)

        # Adversarial tasks (with gradient reversal)
        pooled_adv = grad_reverse(pooled, alpha)
        adv_text_logits = self.adversarial_text_classifier(pooled_adv)
        adv_word_logits = self.adversarial_word_classifier(pooled_adv)

        return difficulty_logits, text_level_logits, word_level_logits, adv_text_logits, adv_word_logits


# ========== 3. Dataset Class ==========
class MultiTaskDataset(torch.utils.data.Dataset):
    """Dataset for multi-task learning"""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulty2id = DIFFICULTY2ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Input text
        input_text = f"Word: {item['target_word']} Context: {item['context']}"

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Main task label
        difficulty_label = self.difficulty2id[item['label']]

        # Auxiliary task labels (absolute prediction)
        text_level_label = item['text_level'] - 1
        word_level_label = item.get('word_level', 1) - 1

        # Adversarial task labels (interval)
        text_interval = get_text_level_interval(item['text_level'])
        word_interval = get_word_level_interval(item.get('word_level', 1))

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'difficulty_label': torch.tensor(difficulty_label, dtype=torch.long),
            'text_level_label': torch.tensor(text_level_label, dtype=torch.long),
            'word_level_label': torch.tensor(word_level_label, dtype=torch.long),
            'text_interval': torch.tensor(text_interval, dtype=torch.long),
            'word_interval': torch.tensor(word_interval, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'difficulty_labels': torch.stack([item['difficulty_label'] for item in batch]),
        'text_level_labels': torch.stack([item['text_level_label'] for item in batch]),
        'word_level_labels': torch.stack([item['word_level_label'] for item in batch]),
        'text_intervals': torch.stack([item['text_interval'] for item in batch]),
        'word_intervals': torch.stack([item['word_interval'] for item in batch])
    }


# ========== 4. Evaluation Functions ==========
def evaluate(model, data_loader, device, alpha=0.1):
    """Evaluate model on given data loader"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulty_labels = batch['difficulty_labels'].to(device)

            difficulty_logits, _, _, _, _ = model(input_ids, attention_mask, alpha=alpha)
            preds = torch.argmax(difficulty_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(difficulty_labels.cpu().numpy())

    return all_labels, all_preds


def detailed_evaluation(all_labels, all_preds, dataset_name):
    """Print detailed evaluation results"""
    target_names = ['简单', '可读', '较难', '困难']

    print(f"\n{'=' * 60}")
    print(f"{dataset_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Macro F1: {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("          Predicted")
    print("         简单  可读  较难  困难")
    for i, name in enumerate(target_names):
        print(f"Actual {name}  {cm[i, 0]:5d} {cm[i, 1]:5d} {cm[i, 2]:5d} {cm[i, 3]:5d}")

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted')
    }


# ========== 5. Main Training Loop ==========
def main():
    args = parse_args()

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    # Load data
    train_data = load_jsonl(args.train_file)
    dev_data = load_jsonl(args.dev_file)
    test_data = load_jsonl(args.test_file)

    print(f"Train: {len(train_data)} samples")
    print(f"Dev: {len(dev_data)} samples")
    print(f"Test: {len(test_data)} samples")

    print(f"\nClass weights: simple=0.3, readable=3.5, difficult=20.0, very_difficult=7.0")

    # Initialize tokenizer and model
    print(f"\nInitializing model: {args.model_name}")
    tokenizer = XLNetTokenizer.from_pretrained(args.model_name)
    model = XLNetMultiTaskWithAdversarial(args.model_name)

    # Create datasets and data loaders
    train_dataset = MultiTaskDataset(train_data, tokenizer, args.max_length)
    dev_dataset = MultiTaskDataset(dev_data, tokenizer, args.max_length)
    test_dataset = MultiTaskDataset(test_data, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss functions
    difficulty_loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    text_level_loss_fn = nn.CrossEntropyLoss()
    word_level_loss_fn = nn.CrossEntropyLoss()
    adv_loss_fn = nn.CrossEntropyLoss()

    print(f"\nLoss weights configuration:")
    print(f"  lambda_text: {args.lambda_text}")
    print(f"  lambda_word: {args.lambda_word}")
    print(f"  lambda_adv_text: {args.lambda_adv_text}")
    print(f"  lambda_adv_word: {args.lambda_adv_word}")
    print(f"\nDynamic adversarial coefficient: linear increase (0→{args.max_alpha}) in first 3 epochs")

    # Check for existing model
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)

    start_epoch = 0
    best_f1 = 0
    best_epoch = 0

    # Try to load existing best model
    if os.path.exists(best_model_path):
        print(f"\nFound existing model at {best_model_path}, loading...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        # Note: In a full implementation, you'd also load best_f1 from a checkpoint file

    # Training loop
    print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0

        # Dynamic adversarial coefficient
        alpha = get_alpha(epoch, args.epochs, args.max_alpha)

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} (α={alpha:.3f})")

        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulty_labels = batch['difficulty_labels'].to(device)
            text_level_labels = batch['text_level_labels'].to(device)
            word_level_labels = batch['word_level_labels'].to(device)
            text_intervals = batch['text_intervals'].to(device)
            word_intervals = batch['word_intervals'].to(device)

            optimizer.zero_grad()

            difficulty_logits, text_level_logits, word_level_logits, adv_text_logits, adv_word_logits = model(
                input_ids, attention_mask, alpha=alpha
            )

            # Main task loss
            loss_main = difficulty_loss_fn(difficulty_logits, difficulty_labels)

            # Auxiliary task losses
            loss_text = text_level_loss_fn(text_level_logits, text_level_labels)
            loss_word = word_level_loss_fn(word_level_logits, word_level_labels)

            # Adversarial task losses
            loss_adv_text = adv_loss_fn(adv_text_logits, text_intervals)
            loss_adv_word = adv_loss_fn(adv_word_logits, word_intervals)

            # Total loss
            loss = (loss_main +
                    args.lambda_text * loss_text +
                    args.lambda_word * loss_word +
                    args.lambda_adv_text * loss_adv_text +
                    args.lambda_adv_word * loss_adv_word)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}, α={alpha:.3f}")

        # Validation
        dev_labels, dev_preds = evaluate(model, dev_loader, device, alpha)
        macro_f1 = f1_score(dev_labels, dev_preds, average='macro')
        print(f"Dev Macro F1: {macro_f1:.4f}")

        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Saved best model, F1: {best_f1:.4f} (Epoch {best_epoch})")

    print(f"\nBest dev result: Epoch {best_epoch}, Macro F1 = {best_f1:.4f}")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_labels, test_preds = evaluate(model, test_loader, device, alpha=args.max_alpha)
    test_results = detailed_evaluation(test_labels, test_preds, "Test Set")

    # Save results
    results = {
        'model': 'xlnet-adversarial-dynamic',
        'best_dev_macro_f1': float(best_f1),
        'best_epoch': best_epoch,
        'test_accuracy': float(test_results['accuracy']),
        'test_macro_f1': float(test_results['macro_f1']),
        'test_weighted_f1': float(test_results['weighted_f1']),
        'label_mapping': DIFFICULTY2ID,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'max_length': args.max_length,
            'lambda_text': args.lambda_text,
            'lambda_word': args.lambda_word,
            'lambda_adv_text': args.lambda_adv_text,
            'lambda_adv_word': args.lambda_adv_word,
            'max_alpha': args.max_alpha
        }
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nTraining complete!")
    print(f"Best dev Macro F1: {best_f1:.4f}")
    print(f"Test Macro F1: {test_results['macro_f1']:.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()