query="Sanduíche de bagel estilo Nova York"

uv run scripts/inspect_run.py eval_data/run_routed_20260315_231352.json --query "$query"

uv run scripts/inspect_ground_truth.py --query "$query" --max-items 10