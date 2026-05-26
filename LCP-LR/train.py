"""
Code for lexical complexity classification using XLNet-base.
Submitted to [Conference Name - Anonymized].
"""

import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import numpy as np
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Lexical Complexity Classification')
    parser.add_argument('--train_file', type=str, default='train.jsonl',
                        help='Training data file')
    parser.add_argument('--dev_file', type=str, default='dev.jsonl',
                        help='Validation data file')
    parser.add_argument('--test_file', type=str, default='test.jsonl',
                        help='Test data file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased',
                        help='Pretrained model name or path')
    return parser.parse_args()


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


LABEL2ID = {'简单': 0, '可读': 1, '较难': 2, '困难': 3}
ID2LABEL = {0: '简单', 1: '可读', 2: '较难', 3: '困难'}

# Class weights based on inverse frequency in training set
CLASS_WEIGHTS = torch.tensor([0.3, 3.5, 20.0, 7.0], dtype=torch.float)


def prepare_data(data):
    """Add input_text and label_id to each item"""
    for item in data:
        item['input_text'] = f"词语：{item['target_word']} 语境：{item['context']}"
        item['label_id'] = LABEL2ID[item['label']]
    return data


class ComplexityDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label_id'], dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()

        loss_fct = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(model.device))
        loss = loss_fct(logits, labels.long())

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        'accuracy': round(accuracy_score(labels, preds), 4),
        'macro_f1': round(f1_score(labels, preds, average='macro'), 4),
        'weighted_f1': round(f1_score(labels, preds, average='weighted'), 4),
    }


def detailed_eval(model, dataset, name, device):
    """Print detailed classification report and confusion matrix"""
    model.eval()
    all_preds, all_labels = [], []

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Macro F1: {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['简单', '可读', '较难', '困难'], digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("         预测: 简单  可读  较难  困难")
    for i, label in enumerate(['简单', '可读', '较难', '困难']):
        print(f"实际 {label}: {cm[i, 0]:4d} {cm[i, 1]:4d} {cm[i, 2]:4d} {cm[i, 3]:4d}")

    return {'macro_f1': f1_score(all_labels, all_preds, average='macro')}


def main():
    args = parse_args()

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    train_data = load_jsonl(args.train_file)
    dev_data = load_jsonl(args.dev_file)
    test_data = load_jsonl(args.test_file)

    print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

    # Prepare data
    train_data = prepare_data(train_data)
    dev_data = prepare_data(dev_data)
    test_data = prepare_data(test_data)

    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = XLNetTokenizer.from_pretrained(args.model_name)
    model = XLNetForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )


    train_dataset = ComplexityDataset(train_data, tokenizer, args.max_length)
    dev_dataset = ComplexityDataset(dev_data, tokenizer, args.max_length)
    test_dataset = ComplexityDataset(test_data, tokenizer, args.max_length)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to='none',
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    final_model_path = os.path.join(args.output_dir, 'final_model')
    if os.path.exists(final_model_path):
        print(f"\nFound existing model at {final_model_path}, loading...")
        model = XLNetForSequenceClassification.from_pretrained(final_model_path)
        tokenizer = XLNetTokenizer.from_pretrained(final_model_path)
        trainer.model = model
    else:
        print("\nTraining model...")
        trainer.train()
        print(f"\nSaving model to {final_model_path}")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

    # Evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    detailed_eval(model, dev_dataset, "Validation Set", device)
    test_results = detailed_eval(model, test_dataset, "Test Set", device)

    results = {
        'model': args.model_name,
        'macro_f1': float(test_results['macro_f1']),
        'label_mapping': LABEL2ID
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Test Macro F1: {test_results['macro_f1']:.4f}")


if __name__ == '__main__':
    main()