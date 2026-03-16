# EmbeddingGemma Contrastive Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune EmbeddingGemma-300m on synthetic (query, item) pairs to close the semantic/cultural gap in retrieval.

**Architecture:** GPT-4o-mini generates ~10k synthetic queries from 5k items (biased toward semantic-gap queries). sentence-transformers `SentenceTransformerTrainer` fine-tunes with `MultipleNegativesRankingLoss`. Evaluation uses the existing routed pipeline on 60 held-out GT queries.

**Tech Stack:** `openai` (data gen), `sentence-transformers` v5.3 + `datasets` (training), `torch` MPS backend

**Spec:** `docs/superpowers/specs/2026-03-16-embedding-contrastive-finetuning-design.md`

---

## Task 1: Data Generation Script

**Files:**
- Create: `scripts/generate_training_data.py`

### Step 1: Write the data generation script

```python
"""Generate synthetic (query, item_text) training pairs via GPT-4o-mini."""
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_items

SYSTEM_PROMPT = """\
You are a Brazilian food delivery search query generator. Given a food item from iFood,
generate search queries in Portuguese that a customer might type to find this item.

For each item, generate TWO queries:
1. "semantic_gap": A conceptual, cultural, or occasion-based query that does NOT reuse the
   item's name, category, or main keywords. Think about WHEN/WHY someone would order this,
   what cuisine or cultural context it belongs to, or what occasion it suits.
   Return null if the item is too generic (e.g., plain salt, cooking oil).
2. "direct": A straightforward query using different vocabulary than the item name.

Respond with JSON: {"semantic_gap": "..." or null, "direct": "..."}

Examples:
- Item: "Poke Tropical | Culinária Japonesa | Delicioso Poke Tropical... | ALIMENTOS_PREPARADOS/PRATOS/POKES_BOWLS"
  → {"semantic_gap": "almoço estilo havaiano", "direct": "tigela de peixe cru com arroz"}
- Item: "Feijoada Magra | Especial do Dia | ... feijão preto e carnes selecionadas | ALIMENTOS_PREPARADOS/PRATOS/FEIJOADAS"
  → {"semantic_gap": "prato típico brasileiro de sábado", "direct": "feijão preto com carnes"}
- Item: "Pimentão Verde Extra | Feira | Compra por peso | FLV/LEGUMES/PIMENTAO"
  → {"semantic_gap": "ingrediente para refogado caseiro", "direct": "legume verde para cozinhar"}
- Item: "Hot tofu | Hot roll | Enrolado empanado e frito com tofu... | ALIMENTOS_PREPARADOS/PRATOS/SUSHIS_SASHIMIS_TEMAKIS"
  → {"semantic_gap": "sushi vegetariano frito", "direct": "hot roll de tofu"}\
"""

OUTPUT_PATH = "data/training_pairs.jsonl"
PROGRESS_PATH = "data/training_pairs_progress.json"
MAX_CONCURRENT = 20
MAX_RETRIES = 3
SAVE_EVERY = 100


async def generate_queries(client: AsyncOpenAI, item_text: str) -> dict | None:
    """Call GPT-4o-mini to generate queries for one item. Returns parsed JSON or None."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item_text},
                ],
                temperature=0.8,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"  FAILED after {MAX_RETRIES} retries: {e}")
                return None


async def process_batch(client: AsyncOpenAI, items: list[dict], done_ids: set[str]) -> list[dict]:
    """Process items with bounded concurrency, skipping already-done items."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    pairs: list[dict] = []

    async def process_one(item: dict):
        if item["item_id"] in done_ids:
            return
        async with semaphore:
            result = await generate_queries(client, item["text"])
        if result is None:
            return
        if result.get("semantic_gap"):
            pairs.append({
                "query": result["semantic_gap"],
                "item_text": item["text"],
                "query_type": "semantic_gap",
            })
        if result.get("direct"):
            pairs.append({
                "query": result["direct"],
                "item_text": item["text"],
                "query_type": "direct",
            })

    tasks = [process_one(item) for item in items]
    # Process in chunks for progress saving
    for chunk_start in range(0, len(tasks), SAVE_EVERY):
        chunk = tasks[chunk_start:chunk_start + SAVE_EVERY]
        await asyncio.gather(*chunk)
        done_count = len(done_ids) + chunk_start + len(chunk)
        print(f"  Progress: {done_count}/{len(items)} items processed, {len(pairs)} pairs so far")

    return pairs


def load_progress() -> tuple[set[str], list[dict]]:
    """Load previously generated pairs for resume support."""
    done_ids: set[str] = set()
    existing_pairs: list[dict] = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                pair = json.loads(line)
                existing_pairs.append(pair)
        # Reconstruct done_ids from existing pairs (item_text is unique per item)
        seen_texts = set()
        for p in existing_pairs:
            seen_texts.add(p["item_text"])
        # We need item_ids, so load items and match
        items = load_items()
        for item in items:
            if item["text"] in seen_texts:
                done_ids.add(item["item_id"])
    return done_ids, existing_pairs


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    items = load_items()
    print(f"Loaded {len(items)} items")

    done_ids, existing_pairs = load_progress()
    if done_ids:
        print(f"Resuming: {len(done_ids)} items already done, {len(existing_pairs)} pairs exist")

    remaining = [item for item in items if item["item_id"] not in done_ids]
    print(f"Generating queries for {len(remaining)} remaining items...")

    new_pairs = await process_batch(client, remaining, done_ids)

    # Write all pairs (existing + new)
    all_pairs = existing_pairs + new_pairs
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Stats
    semantic_count = sum(1 for p in all_pairs if p["query_type"] == "semantic_gap")
    direct_count = sum(1 for p in all_pairs if p["query_type"] == "direct")
    print(f"\nDone! Total pairs: {len(all_pairs)}")
    print(f"  semantic_gap: {semantic_count}")
    print(f"  direct: {direct_count}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 1.1:** Create `scripts/generate_training_data.py` with the code above.

- [ ] **Step 1.2:** Smoke-test with 5 items to verify API calls work.

Run: `OPENAI_API_KEY=<key> uv run python scripts/generate_training_data.py --limit 5`

Wait — the script doesn't support `--limit` yet. Add an optional `--limit N` argument to `main()` for testing:

```python
# Add at the top of main(), after loading items:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0, help="Limit items for testing (0=all)")
args = parser.parse_args()
if args.limit:
    items = items[:args.limit]
```

Actually, fold argparse into the script from the start. Add it cleanly.

Expected: 8-10 JSONL lines in `data/training_pairs.jsonl`, each with `query`, `item_text`, `query_type` fields. Portuguese queries that don't just repeat the item name.

- [ ] **Step 1.3:** Spot-check output quality — read a few pairs and verify the semantic_gap queries are genuinely conceptual.

- [ ] **Step 1.4:** Commit.

```bash
git add scripts/generate_training_data.py
git commit -m "feat: add synthetic training data generation script"
```

---

## Task 2: Training Script

**Files:**
- Create: `scripts/train_embedding.py`

- [ ] **Step 2.1:** Write the training script.

```python
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_PATH, help="Path to training_pairs.jsonl")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output model directory")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
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

    # 2. Build HF dataset with columns matching model prompt names:
    #    "anchor" = query text, "positive" = item text
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
    print(f"Set EMBEDDING_MODEL = \"{args.output}\" in src/config.py to use it.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2:** Commit.

```bash
git add scripts/train_embedding.py
git commit -m "feat: add embedding fine-tuning script with MNRL"
```

---

## Task 3: Run Data Generation

- [ ] **Step 3.1:** Run the full data generation (~5,000 API calls).

```bash
OPENAI_API_KEY=<key> uv run python scripts/generate_training_data.py
```

Expected: ~8,000-10,000 lines in `data/training_pairs.jsonl`. Takes ~10-15 min with 20 concurrent calls.

- [ ] **Step 3.2:** Verify output stats and spot-check quality.

```bash
wc -l data/training_pairs.jsonl
# Check type distribution
python3 -c "
import json
pairs = [json.loads(l) for l in open('data/training_pairs.jsonl')]
types = {}
for p in pairs:
    types[p['query_type']] = types.get(p['query_type'], 0) + 1
print(types)
# Show 5 random semantic_gap examples
import random
semantic = [p for p in pairs if p['query_type'] == 'semantic_gap']
for p in random.sample(semantic, min(5, len(semantic))):
    print(f'  Q: {p[\"query\"]}\n  I: {p[\"item_text\"][:80]}...\n')
"
```

---

## Task 4: Train the Model

- [ ] **Step 4.1:** Run training.

```bash
uv run python scripts/train_embedding.py
```

Expected: ~5-15 min on MPS. Watch for val loss decreasing. If OOM, reduce batch size:
```bash
uv run python scripts/train_embedding.py --batch-size 32
```

- [ ] **Step 4.2:** Verify model output exists.

```bash
ls models/embeddinggemma-finetuned/
```

Expected: `config.json`, `model.safetensors`, `tokenizer.json`, etc.

---

## Task 5: Evaluate

- [ ] **Step 5.1:** Update config to use the fine-tuned model.

In `src/config.py`, change:
```python
EMBEDDING_MODEL = "models/embeddinggemma-finetuned"
```

- [ ] **Step 5.2:** Run evaluation.

```bash
OPENAI_API_KEY=<key> uv run python run_eval.py --evaluate routed
```

- [ ] **Step 5.3:** Compare results against baseline.

Baseline (google/embeddinggemma-300m):
- Overall nDCG@10: 0.669
- Semantic nDCG@10: 0.540
- Keyword nDCG@10: 0.849
- Negative nDCG@10: 0.607

Success criteria:
- Semantic nDCG@10 > 0.540
- Keyword nDCG@10 > 0.807 (no more than 5% regression)
- Negative nDCG@10 > 0.577 (no more than 5% regression)

- [ ] **Step 5.4:** Commit the config change and training artifacts (not the model weights — add `models/` to `.gitignore`).

```bash
echo "models/" >> .gitignore
git add src/config.py .gitignore
git commit -m "feat: switch to fine-tuned EmbeddingGemma for routed pipeline"
```
