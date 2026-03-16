"""Fine-tune EmbeddingGemma-300m with MultipleNegativesRankingLoss."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = "data/training_pairs.jsonl"
OUTPUT_DIR = "models/embeddinggemma-finetuned"
BASE_MODEL = "google/embeddinggemma-300m"


def load_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to training_pairs.jsonl")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output model directory")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze transformer, only train projection heads")
    args = parser.parse_args()

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    # 1. Load data
    pairs = load_pairs(args.input)
    print(f"Loaded {len(pairs)} training pairs")

    # 2. Build HF dataset: "anchor" = query, "positive" = item text
    ds = Dataset.from_dict({
        "anchor": [p["query"] for p in pairs],
        "positive": [p["item_text"] for p in pairs],
    })

    # 3. 90/10 train/val split
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 4. Load model
    model = SentenceTransformer(args.base_model)
    print(f"Model prompts: {list(model.prompts.keys())}")

    # Optionally freeze transformer backbone, only train Dense projection heads
    if args.freeze_backbone:
        for param in model[0].parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Frozen backbone: training {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    # 5. Loss
    loss = losses.MultipleNegativesRankingLoss(model)

    # 6. Training args
    num_train_steps = (len(train_ds) // args.batch_size) * args.epochs
    eval_steps = max(1, num_train_steps // (args.epochs * 4))  # ~4 evals per epoch

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        # Map dataset columns to model prompt templates
        prompts={
            "anchor": "query",
            "positive": "document",
        },
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    # 7. Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
    )

    trainer.train()

    # 8. Save best model
    model.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")
    print(f'Set EMBEDDING_MODEL = "{args.output}" in src/config.py to use it.')


if __name__ == "__main__":
    main()
