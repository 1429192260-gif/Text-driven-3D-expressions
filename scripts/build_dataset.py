import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.param_utils import params_dict_to_vector

DEFAULT_INPUT_PATH = "data/raw_data.json"
DEFAULT_OUTPUT_PATH = "data/train.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw text-expression samples into training format.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    for item in data:
        processed.append({
            "text": item["text"],
            "emotion": item["emotion"],
            "intensity": float(item["intensity"]),
            "param_vector": params_dict_to_vector(item["params"])
        })

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(processed)} samples to {args.output_path}")


if __name__ == "__main__":
    main()
