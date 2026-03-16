# src/eval/build_ground_truth.py
import hashlib
import json
import os
import re
import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.config import (
    PROXY_URL,
    PROXY_KEY,
    OPENAI_API_KEY,
    LLM_MODEL,
    EVAL_DIR,
    GT_BATCH_SIZE,
    GT_ROUND1_TOP_PER_BATCH,
    GT_ROUND1_RUNS,
    GT_ROUND2_RUNS,
)


def get_client() -> OpenAI:
    key = PROXY_KEY or OPENAI_API_KEY
    return OpenAI(api_key=key)


def format_items_for_prompt(items: list[dict]) -> tuple[str, dict[str, str]]:
    """Format items with integer indices. Returns (text, idx_to_id mapping)."""
    lines = []
    idx_to_id: dict[str, str] = {}
    for i, item in enumerate(items):
        idx_to_id[str(i)] = item["item_id"]
        tax = item.get("taxonomy", {})
        tax_str = f"{tax.get('l0', '')}/{tax.get('l1', '')}/{tax.get('l2', '')}"
        line = f"[{i}] {item['name']} | {item['category_name']} | {item['description']} | {tax_str}"
        lines.append(line)
    return "\n".join(lines), idx_to_id


def build_round1_prompt(query: str, items_text: str, top_n: int = 10) -> str:
    return f"""You are evaluating search relevance for a food delivery app (iFood, Brazil).

Query: "{query}"

Below are items from the catalog. Select the {top_n} most relevant items for this query.
Return ONLY a JSON array of the numeric indices from the [brackets], like: [0, 3, 7, ...]

Items:
{items_text}"""


def parse_round1_response(response: str) -> list[str]:
    """Extract list of item IDs from LLM response."""
    # Try direct JSON parse
    text = response.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    # Fallback: find anything that looks like a JSON array
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    return []


def build_round2_prompt(query: str, query_category: str, items_text: str) -> str:
    negation_instruction = ""
    if query_category == "negative":
        negation_instruction = """
IMPORTANT: This is a NEGATION query. The user explicitly does NOT want certain ingredients/attributes.
For each item, check if it violates the negation constraint. Mark violation=true if the item contains
what the user explicitly excluded."""

    return f"""You are evaluating search relevance for a food delivery app (iFood, Brazil).

Query: "{query}"
Query type: {query_category}
{negation_instruction}

Score each item's relevance to the query:
- 3 = highly relevant (exactly what the user wants)
- 2 = relevant (good match, minor differences)
- 1 = marginally relevant (loosely related)
- 0 = irrelevant

Return a JSON array of objects: [{{"id": 0, "score": 0-3, "violation": true/false}}]
Use the numeric index from the [brackets] as the id. Set violation=true ONLY if the item contradicts an explicit negation in the query.

Items:
{items_text}"""


def parse_round2_response(response: str) -> list[dict]:
    """Extract scored items from LLM response."""
    text = response.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: find JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def run_round1(client: OpenAI, query: str, items: list[dict]) -> set[str]:
    """Round 1: coarse filtering. Returns set of candidate item IDs."""
    candidates = set()

    # Split items into batches
    batches = [items[i:i + GT_BATCH_SIZE] for i in range(0, len(items), GT_BATCH_SIZE)]

    for batch in tqdm(batches, desc="  R1 batches", leave=False):
        items_text, idx_to_id = format_items_for_prompt(batch)
        prompt = build_round1_prompt(query, items_text, top_n=GT_ROUND1_TOP_PER_BATCH)

        for run in range(GT_ROUND1_RUNS):
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                )
                indices = parse_round1_response(resp.choices[0].message.content or "")
                # Map integer indices back to real item IDs
                for idx in indices:
                    real_id = idx_to_id.get(str(idx))
                    if real_id:
                        candidates.add(real_id)
            except Exception as e:
                print(f"  Round 1 error (run {run}): {e}")
                time.sleep(2)

    return candidates


def run_round2(
    client: OpenAI,
    query: str,
    query_category: str,
    candidate_items: list[dict],
) -> list[dict]:
    """Round 2: fine-grained scoring. Returns list of {id, score, violation}."""
    items_text, idx_to_id = format_items_for_prompt(candidate_items)
    prompt = build_round2_prompt(query, query_category, items_text)

    all_scores: dict[str, list[int]] = {}
    all_violations: dict[str, list[bool]] = {}

    for run in range(GT_ROUND2_RUNS):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
            )
            scored = parse_round2_response(resp.choices[0].message.content or "")
            for entry in scored:
                idx = str(entry.get("id", ""))
                real_id = idx_to_id.get(idx)
                if not real_id:
                    continue
                score = int(entry.get("score", 0))
                violation = bool(entry.get("violation", False))
                all_scores.setdefault(real_id, []).append(score)
                all_violations.setdefault(real_id, []).append(violation)
        except Exception as e:
            print(f"  Round 2 error (run {run}): {e}")
            time.sleep(2)

    # Compute median score and majority-vote violation
    results = []
    for item_id in all_scores:
        scores = sorted(all_scores[item_id])
        median_score = scores[len(scores) // 2]
        violation = sum(all_violations.get(item_id, [])) > len(all_violations.get(item_id, [])) / 2
        results.append({
            "item_id": item_id,
            "relevance": median_score,
            "violation": violation,
        })

    # Sort by relevance descending
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results


def prepare_round1_jsonl(queries: list[dict], items: list[dict], path: str) -> dict:
    """Prepare JSONL for all Round 1 requests.

    Returns idx_mappings: dict mapping "query_idx:batch_idx" -> idx_to_id.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    item_batches = [items[i:i + GT_BATCH_SIZE] for i in range(0, len(items), GT_BATCH_SIZE)]
    idx_mappings: dict[str, dict] = {}

    with open(path, "w", encoding="utf-8") as f:
        for q_idx, q in enumerate(queries):
            query_text = q["query"]
            for b_idx, batch in enumerate(item_batches):
                items_text, idx_to_id = format_items_for_prompt(batch)
                prompt = build_round1_prompt(query_text, items_text, top_n=GT_ROUND1_TOP_PER_BATCH)
                idx_mappings[f"{q_idx}:{b_idx}"] = idx_to_id
                for run_idx in range(GT_ROUND1_RUNS):
                    request = {
                        "custom_id": f"r1|{q_idx}|{b_idx}|{run_idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": LLM_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 1.0,
                        },
                    }
                    f.write(json.dumps(request) + "\n")

    return idx_mappings


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_registry(registry_path: str) -> dict:
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict, registry_path: str) -> None:
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def _download_batch_results(client: OpenAI, output_file_id: str) -> list[dict]:
    content = client.files.content(output_file_id)
    return [json.loads(line) for line in content.text.strip().split("\n") if line.strip()]


def _poll_until_done(client: OpenAI, batch_id: str, poll_interval: int) -> Any:
    """Poll batch until terminal state. Returns the final batch object."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  Status: {batch.status} | completed: {counts.completed}/{counts.total}, failed: {counts.failed}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_interval)


def _split_jsonl(jsonl_path: str, max_requests_per_chunk: int = 150) -> list[str]:
    """Split a JSONL file into smaller chunk files.

    Returns list of chunk file paths. If the file is small enough,
    returns a single-element list with the original path.
    """
    with open(jsonl_path) as f:
        lines = f.readlines()

    if len(lines) <= max_requests_per_chunk:
        return [jsonl_path]

    base, ext = os.path.splitext(jsonl_path)
    chunk_paths = []
    for i in range(0, len(lines), max_requests_per_chunk):
        chunk = lines[i:i + max_requests_per_chunk]
        chunk_path = f"{base}_chunk{len(chunk_paths)}{ext}"
        with open(chunk_path, "w") as f:
            f.writelines(chunk)
        chunk_paths.append(chunk_path)

    print(f"  Split {len(lines)} requests into {len(chunk_paths)} chunks of ~{max_requests_per_chunk}")
    return chunk_paths


def _submit_single_batch(
    client: OpenAI,
    jsonl_path: str,
    poll_interval: int,
    registry: dict,
    registry_path: str | None,
) -> list[dict]:
    """Submit a single JSONL file as a batch, with registry-based caching."""
    content_hash = _hash_file(jsonl_path)

    # --- Try to reuse an existing batch for this exact content ---
    if content_hash in registry:
        batch_id = registry[content_hash]
        try:
            batch = client.batches.retrieve(batch_id)
            print(f"  Found existing batch {batch_id} for this content (status: {batch.status})")
            if batch.status == "completed":
                print("  Reusing completed batch — downloading results...")
                return _download_batch_results(client, batch.output_file_id)
            elif batch.status in ("failed", "expired", "cancelled"):
                print(f"  Cached batch ended with '{batch.status}' — submitting fresh.")
                del registry[content_hash]
                if registry_path:
                    _save_registry(registry, registry_path)
            else:
                print(f"  Resuming in-progress batch. Polling every {poll_interval}s...")
                batch = _poll_until_done(client, batch_id, poll_interval)
                if batch.status != "completed":
                    raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")
                return _download_batch_results(client, batch.output_file_id)
        except Exception as e:
            print(f"  Could not retrieve cached batch ({e}) — submitting fresh.")
            registry.pop(content_hash, None)
            if registry_path:
                _save_registry(registry, registry_path)

    # --- Submit new batch ---
    print(f"  Uploading batch file: {jsonl_path}")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  File uploaded: {file_obj.id}. Creating batch...")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch created: {batch.id}. Polling every {poll_interval}s...")

    # Save to registry before polling so a crash mid-wait can still be resumed
    if registry_path:
        registry[content_hash] = batch.id
        _save_registry(registry, registry_path)

    batch = _poll_until_done(client, batch.id, poll_interval)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

    print(f"  Downloading results from file: {batch.output_file_id}")
    return _download_batch_results(client, batch.output_file_id)


def submit_batch_and_wait(
    client: OpenAI,
    jsonl_path: str,
    poll_interval: int = 30,
    registry_path: str | None = None,
    max_requests_per_chunk: int = 150,
) -> list[dict]:
    """Upload JSONL file, create batch, poll until complete, return results.

    Large files are automatically split into chunks of max_requests_per_chunk
    requests and submitted sequentially to stay within API token limits.

    If registry_path is given, a JSON registry of {content_hash: batch_id} is
    maintained there.  On each call the JSONL is hashed; if a matching batch
    already exists on the API it is reused (completed → download immediately,
    in-progress → resume polling, failed/expired → remove and resubmit).
    Delete the registry file to force a completely fresh submission.
    """
    chunk_paths = _split_jsonl(jsonl_path, max_requests_per_chunk)
    registry = _load_registry(registry_path) if registry_path else {}

    all_results = []
    for i, chunk_path in enumerate(chunk_paths):
        if len(chunk_paths) > 1:
            print(f"  --- Chunk {i + 1}/{len(chunk_paths)} ---")
        results = _submit_single_batch(client, chunk_path, poll_interval, registry, registry_path)
        all_results.extend(results)

    return all_results


def parse_round1_batch_results(
    results: list[dict],
    idx_mappings: dict,
    queries: list[dict],
) -> dict:
    """Parse Round 1 batch results. Returns dict[query_text -> set[item_id]]."""
    candidates_per_query: dict[str, set] = {q["query"]: set() for q in queries}

    for result in results:
        custom_id = result.get("custom_id", "")
        parts = custom_id.split("|")
        if len(parts) != 4 or parts[0] != "r1":
            continue
        if result.get("error"):
            print(f"  Request {custom_id} failed: {result['error']}")
            continue

        q_idx, b_idx = int(parts[1]), int(parts[2])
        query_text = queries[q_idx]["query"]
        idx_to_id = idx_mappings.get(f"{q_idx}:{b_idx}", {})

        try:
            content = result["response"]["body"]["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            continue

        for idx in parse_round1_response(content):
            real_id = idx_to_id.get(str(idx))
            if real_id:
                candidates_per_query[query_text].add(real_id)

    return candidates_per_query


def prepare_round2_jsonl(
    queries: list[dict],
    candidates_per_query: dict,
    item_lookup: dict,
    path: str,
) -> dict:
    """Prepare JSONL for all Round 2 requests.

    Returns idx_mappings: dict mapping query_idx (str) -> idx_to_id.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idx_mappings: dict[str, dict] = {}

    with open(path, "w", encoding="utf-8") as f:
        for q_idx, q in enumerate(queries):
            query_text = q["query"]
            query_cat = q["category"]
            candidate_ids = candidates_per_query.get(query_text, set())
            candidate_items = [item_lookup[cid] for cid in candidate_ids if cid in item_lookup]
            if not candidate_items:
                continue

            items_text, idx_to_id = format_items_for_prompt(candidate_items)
            prompt = build_round2_prompt(query_text, query_cat, items_text)
            idx_mappings[str(q_idx)] = idx_to_id

            for run_idx in range(GT_ROUND2_RUNS):
                request = {
                    "custom_id": f"r2|{q_idx}|{run_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 1.0,
                    },
                }
                f.write(json.dumps(request) + "\n")

    return idx_mappings


def parse_round2_batch_results(
    results: list[dict],
    idx_mappings: dict,
    queries: list[dict],
) -> dict:
    """Parse Round 2 batch results. Returns ground truth dict."""
    all_scores: dict[str, dict[str, list[int]]] = {}
    all_violations: dict[str, dict[str, list[bool]]] = {}

    for result in results:
        custom_id = result.get("custom_id", "")
        parts = custom_id.split("|")
        if len(parts) != 3 or parts[0] != "r2":
            continue
        if result.get("error"):
            print(f"  Request {custom_id} failed: {result['error']}")
            continue

        q_idx = int(parts[1])
        query_text = queries[q_idx]["query"]
        idx_to_id = idx_mappings.get(str(q_idx), {})

        try:
            content = result["response"]["body"]["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            continue

        for entry in parse_round2_response(content):
            real_id = idx_to_id.get(str(entry.get("id", "")))
            if not real_id:
                continue
            score = int(entry.get("score", 0))
            violation = bool(entry.get("violation", False))
            all_scores.setdefault(query_text, {}).setdefault(real_id, []).append(score)
            all_violations.setdefault(query_text, {}).setdefault(real_id, []).append(violation)

    ground_truth = {}
    for q in queries:
        query_text = q["query"]
        q_scores = all_scores.get(query_text, {})
        q_violations = all_violations.get(query_text, {})

        scored_items = []
        for item_id, scores in q_scores.items():
            sorted_scores = sorted(scores)
            median_score = sorted_scores[len(sorted_scores) // 2]
            violations = q_violations.get(item_id, [])
            violation = sum(violations) > len(violations) / 2
            scored_items.append({"item_id": item_id, "relevance": median_score, "violation": violation})

        scored_items.sort(key=lambda x: x["relevance"], reverse=True)
        ground_truth[query_text] = scored_items

    return ground_truth


def build_ground_truth(items: list[dict], queries: list[dict]) -> dict:
    """Build complete ground truth for all queries using the OpenAI Batch API."""
    client = get_client()
    item_lookup = {item["item_id"]: item for item in items}
    tmp_dir = os.path.join(EVAL_DIR, "batch_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # --- Round 1: coarse candidate selection ---
    r1_path = os.path.join(tmp_dir, "round1.jsonl")
    print("Preparing Round 1 JSONL...")
    r1_mappings = prepare_round1_jsonl(queries, items, r1_path)
    with open(r1_path) as f:
        n_r1 = sum(1 for line in f if line.strip())
    print(f"  {n_r1} requests prepared")

    registry = os.path.join(tmp_dir, "batch_registry.json")
    print("Submitting Round 1 batch...")
    r1_results = submit_batch_and_wait(client, r1_path, registry_path=registry)
    print(f"  Round 1 complete: {len(r1_results)} results received")

    candidates_per_query = parse_round1_batch_results(r1_results, r1_mappings, queries)
    for q in queries:
        n = len(candidates_per_query.get(q["query"], set()))
        print(f"  '{q['query']}': {n} candidates")

    # --- Round 2: fine-grained scoring ---
    r2_path = os.path.join(tmp_dir, "round2.jsonl")
    print("Preparing Round 2 JSONL...")
    r2_mappings = prepare_round2_jsonl(queries, candidates_per_query, item_lookup, r2_path)
    with open(r2_path) as f:
        n_r2 = sum(1 for line in f if line.strip())
    print(f"  {n_r2} requests prepared")

    print("Submitting Round 2 batch...")
    r2_results = submit_batch_and_wait(client, r2_path, registry_path=registry)
    print(f"  Round 2 complete: {len(r2_results)} results received")

    ground_truth = parse_round2_batch_results(r2_results, r2_mappings, queries)
    return ground_truth


def save_ground_truth(ground_truth: dict, path: str | None = None):
    """Save ground truth to JSON."""
    if path is None:
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, "ground_truth.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"Ground truth saved to {path}")


def load_ground_truth(path: str | None = None) -> dict:
    """Load ground truth from JSON."""
    if path is None:
        path = os.path.join(EVAL_DIR, "ground_truth.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
