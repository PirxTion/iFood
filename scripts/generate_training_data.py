"""Generate synthetic (query, item_text) training pairs via GPT-4o-mini."""
import argparse
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
MAX_CONCURRENT = 20
MAX_RETRIES = 3


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


async def process_items(client: AsyncOpenAI, items: list[dict], done_texts: set[str]) -> list[dict]:
    """Process items with bounded concurrency, skipping already-done items."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    pairs: list[dict] = []
    processed = 0

    async def process_one(item: dict):
        nonlocal processed
        if item["text"] in done_texts:
            return
        async with semaphore:
            result = await generate_queries(client, item["text"])
        processed += 1
        if processed % 100 == 0:
            print(f"  Progress: {processed}/{len(items)} items, {len(pairs)} pairs")
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

    await asyncio.gather(*[process_one(item) for item in items])
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


async def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training pairs")
    parser.add_argument("--limit", type=int, default=0, help="Limit items for testing (0=all)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
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

    print(f"Generating queries for {len(remaining)} remaining items...")
    new_pairs = await process_items(client, remaining, done_texts)

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
    asyncio.run(main())
