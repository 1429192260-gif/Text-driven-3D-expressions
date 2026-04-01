import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.param_utils import params_dict_to_vector

INPUT_PATH = "data/raw_data.json"
OUTPUT_PATH = "data/train.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

processed = []
for item in data:
    processed.append({
        "text": item["text"],
        "emotion": item["emotion"],
        "intensity": float(item["intensity"]),
        "param_vector": params_dict_to_vector(item["params"])
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)

print(f"Saved {len(processed)} samples to {OUTPUT_PATH}")