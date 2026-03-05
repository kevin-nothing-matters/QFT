"""Check structure of per-tree JSON to fix loader."""
import json, os
fname = os.path.expanduser("~/qft/results/depth8_tree000.json")
with open(fname) as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys())[:10])
print("Type of first value:", type(list(data.values())[0]))
first_val = list(data.values())[0]
if isinstance(first_val, list):
    print("List length:", len(first_val))
    print("First 5 values:", first_val[:5])
elif isinstance(first_val, dict):
    print("Dict keys:", list(first_val.keys()))
else:
    print("Value:", first_val)
