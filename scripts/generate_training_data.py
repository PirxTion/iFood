"""Generate synthetic (query, item_text) training pairs via GPT-4o-mini Batch API."""
import argparse
import hashlib
import json
import os
import sys
import time

from openai import OpenAI

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
TMP_DIR = "data/batch_tmp"
REGISTRY_PATH = os.path.join(TMP_DIR, "training_batch_registry.json")
MAX_REQUESTS_PER_CHUNK = 50000  # Send everything in one batch


def get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    return OpenAI(api_key=key)


def prepare_batch_jsonl(items: list[dict], path: str) -> None:
    """Write one Batch API request per item to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            request = {
                "custom_id": f"train|{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-5-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["text"]},
                    ],
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(request) + "\n")
    print(f"Prepared {len(items)} requests → {path}")


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_registry(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def _download_batch_results(client: OpenAI, output_file_id: str) -> list[dict]:
    content = client.files.content(output_file_id)
    return [json.loads(line) for line in content.text.strip().split("\n") if line.strip()]


def _poll_until_done(client: OpenAI, batch_id: str, poll_interval: int = 30):
    """Poll batch until terminal state."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  Status: {batch.status} | completed: {counts.completed}/{counts.total}, failed: {counts.failed}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_interval)


def _split_jsonl(jsonl_path: str, max_per_chunk: int) -> list[str]:
    """Split a JSONL file into smaller chunks if needed."""
    with open(jsonl_path) as f:
        lines = f.readlines()
    if len(lines) <= max_per_chunk:
        return [jsonl_path]
    base, ext = os.path.splitext(jsonl_path)
    chunk_paths = []
    for i in range(0, len(lines), max_per_chunk):
        chunk = lines[i:i + max_per_chunk]
        chunk_path = f"{base}_chunk{len(chunk_paths)}{ext}"
        with open(chunk_path, "w") as f:
            f.writelines(chunk)
        chunk_paths.append(chunk_path)
    print(f"  Split {len(lines)} requests into {len(chunk_paths)} chunks of ~{max_per_chunk}")
    return chunk_paths


def _submit_single_batch(
    client: OpenAI, jsonl_path: str, registry: dict
) -> list[dict]:
    """Submit a single JSONL file as a batch, with registry-based caching."""
    content_hash = _hash_file(jsonl_path)

    # Try to reuse an existing batch for this content
    if content_hash in registry:
        batch_id = registry[content_hash]
        try:
            batch = client.batches.retrieve(batch_id)
            print(f"  Found existing batch {batch_id} (status: {batch.status})")
            if batch.status == "completed":
                print("  Reusing completed batch — downloading results...")
                return _download_batch_results(client, batch.output_file_id)
            elif batch.status in ("failed", "expired", "cancelled"):
                print(f"  Cached batch ended with '{batch.status}' — submitting fresh.")
                del registry[content_hash]
                _save_registry(registry, REGISTRY_PATH)
            else:
                print("  Resuming in-progress batch...")
                batch = _poll_until_done(client, batch_id)
                if batch.status != "completed":
                    raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")
                return _download_batch_results(client, batch.output_file_id)
        except Exception as e:
            print(f"  Could not retrieve cached batch ({e}) — submitting fresh.")
            registry.pop(content_hash, None)
            _save_registry(registry, REGISTRY_PATH)

    # Submit new batch
    print(f"  Uploading: {jsonl_path}")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  File uploaded: {file_obj.id}. Creating batch...")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",  # OpenAI only supports "24h" currently
    )
    print(f"  Batch created: {batch.id}")

    # Save to registry before polling (resume on crash)
    registry[content_hash] = batch.id
    _save_registry(registry, REGISTRY_PATH)

    batch = _poll_until_done(client, batch.id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

    return _download_batch_results(client, batch.output_file_id)


def submit_batch_and_wait(client: OpenAI, jsonl_path: str) -> list[dict]:
    """Submit batch (with chunking and registry caching), return all results."""
    chunk_paths = _split_jsonl(jsonl_path, MAX_REQUESTS_PER_CHUNK)
    registry = _load_registry(REGISTRY_PATH)

    all_results = []
    for i, chunk_path in enumerate(chunk_paths):
        if len(chunk_paths) > 1:
            print(f"  --- Chunk {i + 1}/{len(chunk_paths)} ---")
        results = _submit_single_batch(client, chunk_path, registry)
        all_results.extend(results)
    return all_results


def parse_batch_results(results: list[dict], items: list[dict]) -> list[dict]:
    """Parse batch results into training pairs."""
    pairs = []
    failed = 0
    for result in results:
        custom_id = result.get("custom_id", "")
        parts = custom_id.split("|")
        if len(parts) != 2 or parts[0] != "train":
            continue
        if result.get("error"):
            failed += 1
            continue

        item_idx = int(parts[1])
        item_text = items[item_idx]["text"]

        try:
            content = result["response"]["body"]["choices"][0]["message"]["content"] or ""
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            failed += 1
            continue

        if parsed.get("semantic_gap"):
            pairs.append({
                "query": parsed["semantic_gap"],
                "item_text": item_text,
                "query_type": "semantic_gap",
            })
        if parsed.get("direct"):
            pairs.append({
                "query": parsed["direct"],
                "item_text": item_text,
                "query_type": "direct",
            })

    if failed:
        print(f"  {failed} requests failed or unparseable")
    return pairs


def load_existing() -> tuple[set[str], list[dict]]:
    """Load previously generated pairs for resume support."""
    existing_pairs: list[dict] = []
    done_texts: set[str] = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                pair = json.loads(line)
                existing_pairs.append(pair)
                done_texts.add(pair["item_text"])
    return done_texts, existing_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training pairs via Batch API")
    parser.add_argument("--limit", type=int, default=0, help="Limit items for testing (0=all)")
    args = parser.parse_args()

    client = get_client()
    items = load_items()
    if args.limit:
        items = items[:args.limit]
    print(f"Loaded {len(items)} items")

    done_texts, existing_pairs = load_existing()
    if done_texts:
        print(f"Resuming: {len(done_texts)} items already done, {len(existing_pairs)} pairs exist")

    remaining = [item for item in items if item["text"] not in done_texts]
    if not remaining:
        print("All items already processed!")
        return

    # Prepare and submit batch
    jsonl_path = os.path.join(TMP_DIR, "training_requests.jsonl")
    prepare_batch_jsonl(remaining, jsonl_path)

    print(f"Submitting batch for {len(remaining)} items...")
    results = submit_batch_and_wait(client, jsonl_path)
    print(f"Batch complete: {len(results)} results received")

    # Parse results
    new_pairs = parse_batch_results(results, remaining)

    # Write all pairs
    all_pairs = existing_pairs + new_pairs
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    semantic_count = sum(1 for p in all_pairs if p["query_type"] == "semantic_gap")
    direct_count = sum(1 for p in all_pairs if p["query_type"] == "direct")
    print(f"\nDone! Total pairs: {len(all_pairs)}")
    print(f"  semantic_gap: {semantic_count}")
    print(f"  direct: {direct_count}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
