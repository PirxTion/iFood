query="temaki sem salmão e sem cream cheese"

uv run scripts/inspect_run.py eval_data/run_dense_20260315_162558.json --query "$query"

uv run scripts/inspect_ground_truth.py --query "$query" --max-items 10