# Plan: Convert Ground Truth Generation to OpenAI Batch API

## Context
The current ground truth builder makes ~1,200+ synchronous API calls (60 queries × 10 batches × 2 runs for R1, plus 60 × 3 runs for R2). This is slow and rate-limit-prone. The OpenAI Batch API offers 50% cost savings and higher rate limits by processing requests asynchronously.

**Key constraint**: Round 2 depends on Round 1 results (candidates), so we need **2 sequential batch submissions**.

## Changes

### File: `src/eval/build_ground_truth.py`

**Keep unchanged**: `format_items_for_prompt`, `build_round1_prompt`, `build_round2_prompt`, `parse_round1_response`, `parse_round2_response`, `save_ground_truth`, `load_ground_truth`, `get_client`.

**Add new functions:**

1. **`prepare_round1_jsonl(queries, items, path)`** — For each query × batch × run, generate a JSONL line with:
   - `custom_id`: encoded as `r1|{query_idx}|{batch_idx}|{run_idx}`
   - Standard chat completions body with the existing prompt builders
   - Also save the `idx_to_id` mappings to a sidecar JSON (needed to map indices back to real IDs)

2. **`submit_batch_and_wait(client, jsonl_path, poll_interval=30)`** — Upload file, create batch, poll status until completed/failed, download and return results as list of dicts.

3. **`parse_round1_batch_results(results, idx_mappings, queries)`** — For each result, decode `custom_id`, parse response with existing `parse_round1_response`, map indices back to real IDs using the sidecar mappings. Returns `dict[query_text → set[item_id]]`.

4. **`prepare_round2_jsonl(queries, candidates_per_query, item_lookup, path)`** — For each query × run, generate JSONL with:
   - `custom_id`: `r2|{query_idx}|{run_idx}`
   - Save idx_to_id mappings sidecar

5. **`parse_round2_batch_results(results, idx_mappings, queries)`** — Decode, parse with existing `parse_round2_response`, aggregate scores with median + majority-vote violation (same logic as current `run_round2`). Returns final ground truth dict.

6. **`build_ground_truth(items, queries)`** — New orchestrator:
   - Prepare R1 JSONL → submit batch → wait → parse R1 results
   - Prepare R2 JSONL from R1 candidates → submit batch → wait → parse R2 results
   - Return ground truth dict

**Temp files** stored in `eval_data/batch_tmp/` (created/cleaned automatically).

### File: `scripts/run_ground_truth.py`
No changes needed — it already calls `build_ground_truth()`.

## Verification
1. Run `uv run python scripts/run_ground_truth.py` in tmux
2. Check that R1 batch submits and completes (printed status updates)
3. Check that R2 batch submits and completes
4. Verify `eval_data/ground_truth.json` is generated with scored items per query
5. Spot-check a few queries for reasonable candidate counts and scores
