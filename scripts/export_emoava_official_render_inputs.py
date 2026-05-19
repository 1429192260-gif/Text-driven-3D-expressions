import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


DEFAULT_METHODS = {
    "gold": None,
    "baseline": "outputs/when_words_smile_repro/test_parallel_full.pt",
    "v4_global": "outputs/when_words_smile_prior_v4/test_uncertainty_prior_fusion_no_prior_branch.pt",
    "v5_full": "outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion.pt",
    "v6_frame": "outputs/when_words_smile_prior_v6/test_mixture_gate.pt",
    "v6_sample": "outputs/when_words_smile_prior_v6/test_mixture_gate_sample.pt",
}


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_cases(raw: str) -> list[int]:
    cases = []
    for item in raw.replace(" ", "").split(","):
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("Case ids are 1-based and must be positive.")
        cases.append(value)
    if not cases:
        raise ValueError("No valid cases were provided.")
    return cases


def maybe_subsample(seq: np.ndarray, max_frames: int, frame_stride: int) -> np.ndarray:
    if frame_stride > 1:
        seq = seq[::frame_stride]
    if max_frames > 0 and len(seq) > max_frames:
        idx = np.linspace(0, len(seq) - 1, max_frames).astype(np.int64)
        seq = seq[idx]
    return np.asarray(seq, dtype=np.float32)


def load_prediction(repo: Path, method: str) -> torch.Tensor | None:
    rel_path = DEFAULT_METHODS[method]
    if rel_path is None:
        return None
    path = repo / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction for method `{method}`: {path}")
    return torch.load(path, map_location="cpu")


def write_method_pickle(
    repo: Path,
    output_dir: Path,
    method: str,
    case_ids: list[int],
    gold_exps: list[np.ndarray],
    max_frames: int,
    frame_stride: int,
) -> dict:
    prediction = load_prediction(repo, method)
    sequences = []

    for case_id in case_ids:
        idx = case_id - 1
        gold_seq = np.asarray(gold_exps[idx], dtype=np.float32)
        if method == "gold":
            seq = gold_seq
        else:
            assert prediction is not None
            seq = prediction[idx, : len(gold_seq)].detach().cpu().numpy().astype(np.float32)
        if seq.ndim != 2 or seq.shape[1] != 53:
            raise ValueError(f"Method `{method}` case {case_id} has invalid shape {seq.shape}; expected [T, 53].")
        sequences.append(maybe_subsample(seq, max_frames=max_frames, frame_stride=frame_stride))

    out_path = output_dir / f"{method}_selected_exps.pkl"
    with out_path.open("wb") as f:
        pickle.dump(sequences, f)

    display_path = out_path.resolve()
    try:
        display_path = display_path.relative_to(repo.resolve())
    except ValueError:
        pass

    return {
        "method": method,
        "source": DEFAULT_METHODS[method] if DEFAULT_METHODS[method] is not None else "external/EmoAva/dataset/test_stage1_exps.pkl",
        "exp_path": str(display_path.as_posix()),
        "num_sequences": len(sequences),
        "sequence_lengths": [int(len(x)) for x in sequences],
    }


def write_manifest(
    output_dir: Path,
    methods: list[dict],
    case_ids: list[int],
    texts: list[str],
    gold_exps: list[np.ndarray],
    max_frames: int,
    frame_stride: int,
):
    records = []
    for row, case_id in enumerate(case_ids):
        idx = case_id - 1
        records.append(
            {
                "render_row": row,
                "case_id_1based": case_id,
                "case_index_0based": idx,
                "text": texts[idx],
                "original_gold_length": int(len(gold_exps[idx])),
            }
        )

    manifest = {
        "note": "These pickle files are official EmoAva visualize.py compatible list-of-[T,53] expression sequences.",
        "case_ids_are_1based": True,
        "max_frames": int(max_frames),
        "frame_stride": int(frame_stride),
        "records": records,
        "methods": methods,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Official 3D Render Input Manifest",
        "",
        "These files can be rendered by the official EmoAva/WhenWordsSmile DECA-FLAME visualization backend.",
        "",
        "## Cases",
        "",
        "| Render Row | Case ID | Original Length | Text |",
        "|---:|---:|---:|---|",
    ]
    for item in records:
        safe_text = item["text"].replace("|", "/")
        lines.append(
            f"| {item['render_row']} | {item['case_id_1based']} | {item['original_gold_length']} | {safe_text} |"
        )

    lines.extend(["", "## Methods", "", "| Method | Input PKL | Source | Sequence Lengths |", "|---|---|---|---|"])
    for item in methods:
        lines.append(
            f"| {item['method']} | `{item['exp_path']}` | `{item['source']}` | `{item['sequence_lengths']}` |"
        )

    lines.extend(
        [
            "",
            "## Server Render Command Template",
            "",
            "Run one command per method after installing the official visualization dependencies:",
            "",
            "```bash",
            "python scripts/render_emoava_official_deca.py \\",
            "  --exp-path outputs/emoava_official_render_inputs/v6_showcase/v6_sample_selected_exps.pkl \\",
            "  --output-dir outputs/emoava_official_render_videos/v6_sample \\",
            "  --device cuda \\",
            "  --geometry-only \\",
            "  --no-detail",
            "```",
        ]
    )
    (output_dir / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Export selected EmoAva prediction tensors to official visualize.py-compatible .pkl files."
    )
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", default="gold,baseline,v6_sample")
    parser.add_argument("--cases", default="967,354,295,428,84")
    parser.add_argument("--output-dir", default="outputs/emoava_official_render_inputs/v6_showcase")
    parser.add_argument("--max-frames", type=int, default=0, help="0 keeps the original sequence length.")
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")

    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / args.dataset_dir
    text_path = dataset_dir / f"{args.split}_stage1_text.pkl"
    exp_path = dataset_dir / f"{args.split}_stage1_exps.pkl"
    texts = load_pickle(text_path)
    gold_exps = [np.asarray(x, dtype=np.float32) for x in load_pickle(exp_path)]

    case_ids = parse_cases(args.cases)
    for case_id in case_ids:
        if case_id > len(texts) or case_id > len(gold_exps):
            raise IndexError(f"Case {case_id} is out of range for split `{args.split}`.")

    methods = []
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in [x.strip() for x in args.methods.split(",") if x.strip()]:
        if method not in DEFAULT_METHODS:
            known = ", ".join(DEFAULT_METHODS)
            raise KeyError(f"Unknown method `{method}`. Available methods: {known}")
        methods.append(
            write_method_pickle(
                repo=repo,
                output_dir=output_dir,
                method=method,
                case_ids=case_ids,
                gold_exps=gold_exps,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
            )
        )

    write_manifest(
        output_dir=output_dir,
        methods=methods,
        case_ids=case_ids,
        texts=texts,
        gold_exps=gold_exps,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    print(output_dir / "manifest.md")


if __name__ == "__main__":
    main()
