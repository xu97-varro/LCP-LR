# Data Format

## Files

data/
├── sample.jsonl # Example data
└── README.md


## Format

Each line is a JSON object:

```json
{
  "paragraph_id": "188_12",
  "text_level": 7,
  "target_word": "学者",
  "context": "方：当然有。早期的有鲁迅的作品...",
  "label": "简单",
  "word_level": 5
}

See sample.jsonl for examples.

The complete dataset (100,917 instances, 1,401 texts) is available upon request.