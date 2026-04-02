import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT_ROOT = "data/微博文本情感分析数据-数据集"
DEFAULT_OUTPUT_PATH = "data/weibo_cleaned.json"
DEFAULT_PREVIEW_PATH = "docs/weibo_dataset_preview.md"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean segmented weibo emotion files into plain-text corpus and preview samples."
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--preview-path", default=DEFAULT_PREVIEW_PATH)
    parser.add_argument("--sample-per-label", type=int, default=20)
    return parser.parse_args()


def clean_segmented_line(line: str) -> str:
    tokens = []
    for part in line.strip().split():
        if not part:
            continue
        token = part.rsplit("/", 1)[0] if "/" in part else part
        tokens.append(token)

    text = "".join(tokens)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"…{2,}", "...", text)
    return text.strip()


def find_label_files(input_root: Path):
    files = []
    for path in sorted(input_root.rglob("*_simplifyweibo.txt")):
        label = path.stem.split("_", 1)[0]
        files.append((label, path))
    return files


def build_records(label_files):
    records = []
    grouped = {}
    for label, path in label_files:
        grouped.setdefault(label, [])
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                text = clean_segmented_line(raw)
                if not text:
                    continue
                item = {
                    "source": "weibo_simplify",
                    "label_id": label,
                    "text": text,
                    "source_file": str(path).replace("\\", "/"),
                    "line_no": idx,
                }
                records.append(item)
                grouped[label].append(item)
    return records, grouped


def write_preview(preview_path: Path, grouped, sample_per_label: int):
    tentative_mapping = {
        "0": "偏正向/高兴，含明显赞赏、兴趣、轻松表达",
        "1": "偏愤怒/厌恶，含抱怨、批评、攻击性表达",
        "2": "偏悲伤/消极，含无奈、心酸、哀悼、低落表达",
        "3": "偏中性或混合正向，需进一步人工复核",
    }

    lines = [
        "# 微博情感数据集清洗预览",
        "",
        "该文件由 `scripts/prepare_weibo_corpus.py` 自动生成。",
        "",
        "## 标签统计与初步判断",
        "",
        "| label_id | count | tentative_mapping |",
        "| --- | ---: | --- |",
    ]

    for label in sorted(grouped.keys(), key=lambda x: int(x)):
        lines.append(
            f"| {label} | {len(grouped[label])} | {tentative_mapping.get(label, '待判断')} |"
        )

    lines.extend(["", "## 每类抽样样本", ""])

    for label in sorted(grouped.keys(), key=lambda x: int(x)):
        lines.append(f"### label {label}")
        lines.append("")
        for item in grouped[label][:sample_per_label]:
            lines.append(f"- {item['text']}")
        lines.append("")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    label_files = find_label_files(input_root)
    if not label_files:
        raise FileNotFoundError(f"No *_simplifyweibo.txt files found under: {input_root}")

    records, grouped = build_records(label_files)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    write_preview(Path(args.preview_path), grouped, args.sample_per_label)

    print(f"Saved {len(records)} cleaned records to {output_path}")
    for label in sorted(grouped.keys(), key=lambda x: int(x)):
        print(f"label {label}: {len(grouped[label])}")


if __name__ == "__main__":
    main()
