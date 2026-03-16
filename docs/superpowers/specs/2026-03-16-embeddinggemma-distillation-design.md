# Distill text-embedding-3-large → EmbeddingGemma-300M

## Goal

Fine-tune `google/embeddinggemma-300m` (300M params, 768d) so its item embeddings approximate `text-embedding-3-large@768d` embeddings, then use it as a drop-in local replacement in the dense retrieval pipeline.

## Motivation

- Assignment requires fine-tuning at least one model component
- Removes OpenAI embedding API dependency (latency, cost, rate limits)
- Teacher embeddings already cached — zero additional API calls needed
- EmbeddingGemma is small enough to train locally on Mac MPS GPU

## Training

- **Teacher:** 5,000 item embeddings from `text-embedding-3-large@768d`, cached in `eval_data/embedding_cache.pkl`
- **Student:** `google/embeddinggemma-300m` (full fine-tune, no frozen layers)
- **Loss:** Cosine embedding loss: `loss = 1 - cosine_similarity(student_emb, teacher_emb)`, averaged over batch
- **Data:** 5,000 item texts only (no queries — rely on pretrained query encoding)
- **Device:** MPS (Apple Silicon GPU), fallback to CPU
- **Hyperparameters:**
  - Epochs: 5
  - Batch size: 32 (reduce to 8-16 if MPS OOM; use gradient accumulation to maintain effective batch size)
  - Optimizer: AdamW
  - Learning rate: 2e-5 with linear warmup (10% of steps)
  - Weight decay: 0.01
- **Monitoring:** Log mean loss per epoch; save checkpoint after each epoch

## Data Pipeline

1. Load item texts from `data/5k_items.csv` via existing `load_items()`
2. Load teacher embeddings from `eval_data/embedding_cache.pkl` using the same `_cache_key()` function from `src.retrieval.dense_retriever` (keys are SHA256 hashes of `"text-embedding-3-large:d768:{text}"`)
3. Create a PyTorch Dataset yielding `(item_text, teacher_embedding)` pairs
4. Standard DataLoader with shuffle

## Model Architecture Note

EmbeddingGemma-300M's internal pipeline: Gemma3TextModel (303M) → MeanPooling → Dense(768→3072) → Dense(3072→768) → Normalize. All layers are fine-tuned.

## Prompt Templates

EmbeddingGemma ships with asymmetric prompt templates (`"query: ..."` for queries, `"title: none | text: ..."` for docs). Our vanilla baseline (NDCG 0.491) was measured **without** these prompts. For consistency, the distilled model also trains and infers without prompt templates — both sides encode raw text. This is a conscious simplification; testing with prompts is a possible follow-up.

## New Files

- `src/training/distill_embeddings.py` — training script
  - Loads teacher embeddings from cache
  - Loads item texts via `src.data_loader`
  - Fine-tunes EmbeddingGemma with cosine loss
  - Logs loss per epoch to stdout
  - Saves model to `models/embeddinggemma-distilled/`

## Integration

The distilled model is a standard sentence-transformers model saved to disk. To use it:
- Set `EMBEDDING_MODEL = "models/embeddinggemma-distilled"` in config
- `DenseRetriever` already supports local models (any path with `/` triggers local mode)
- The `_is_local_model()` check sees the `/` in the path and uses `SentenceTransformer(model)`
- First eval run will recompute all 5,000 embeddings (new cache keys due to different model name)

## Housekeeping

- Add `models/` to `.gitignore` (distilled model is ~1.2 GB)

## Evaluation

Run existing eval pipeline with the distilled model:
- `uv run python run_eval.py --evaluate dense` — compare against vanilla EmbeddingGemma (NDCG 0.491) and te3-large@768d (NDCG 0.595)
- `uv run python run_eval.py --evaluate routed` — full pipeline comparison

## Success Criteria

- NDCG@10 meaningfully above vanilla EmbeddingGemma-300M (0.491)
- Ideally approaching te3-large@768d (0.595)
- Honest gap analysis regardless of outcome — the experiment itself satisfies the assignment's fine-tuning requirement

## Risks

- **Query generalization:** Student trains on item texts only (5,000). At inference, unseen queries are encoded by the student. Query embeddings may not land in the right subspace. Mitigated by EmbeddingGemma's pretrained query understanding.
- **Overfitting:** 300M params on 5,000 examples is a high parameter-to-example ratio. Mitigated by low learning rate (2e-5), few epochs (5), and weight decay. If results degrade, freezing the Transformer backbone and training only the Dense projection layers (~4.7M params) is a fallback.
- **Catastrophic forgetting:** Full fine-tune may degrade general multilingual knowledge. Mitigated by low learning rate and few epochs.
- **MPS compatibility:** Some PyTorch ops may not work on MPS. Fallback to CPU if needed.
