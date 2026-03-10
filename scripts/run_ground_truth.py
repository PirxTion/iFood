"""Build LLM-judged ground truth for all queries."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure PROXY_KEY is set
if not os.environ.get("PROXY_KEY"):
    key_path = os.path.join(os.path.dirname(__file__), "..", "key.md")
    # Quick extraction from key.md
    if os.path.exists(key_path):
        with open(key_path) as f:
            for line in f:
                if line.startswith("PROXY_KEY"):
                    os.environ["OPENAI_API_KEY"] = line.split('"')[1]
                    break

from src.data_loader import load_items, load_queries
from src.eval.build_ground_truth import build_ground_truth, save_ground_truth


def main():
    print("Loading data...")
    items = load_items()
    queries = load_queries()
    random.seed(42)
    random.shuffle(items)
    random.shuffle(queries)
    print(f"Loaded {len(items)} items, {len(queries)} queries (shuffled)")

    print("Building ground truth (this will take a while)...")
    gt = build_ground_truth(items, queries)

    save_ground_truth(gt)
    print("Done!")


if __name__ == "__main__":
    main()
