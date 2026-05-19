import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from emoava_numpy_compat import patch_numpy_legacy_aliases


def patch_torch_pickle_load(map_location: str):
    """Allow CUDA-saved tensors inside pickle files to be loaded on CPU too."""

    def _load_from_bytes(buffer):
        return torch.load(io.BytesIO(buffer), map_location=map_location)

    torch.storage._load_from_bytes = _load_from_bytes


def load_pickle(path: Path, map_location: str):
    patch_torch_pickle_load(map_location)
    with path.open("rb") as f:
        return pickle.load(f)


def load_manifest_records(exp_path: Path) -> list[dict] | None:
    manifest_path = exp_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("records")


def import_official_deca(emoava_src: Path):
    patch_numpy_legacy_aliases()
    sys.path.insert(0, str(emoava_src))
    from decalib.deca import DECA
    from decalib.utils.config import get_cfg_defaults

    return DECA, get_cfg_defaults


def make_code(default_code: dict, frame: np.ndarray, device: torch.device, fix_cam: bool = True) -> dict:
    frame_tensor = torch.as_tensor(frame, dtype=torch.float32, device=device)
    exp = frame_tensor[:50].view(1, -1)
    jaw = frame_tensor[50:53]
    pose = torch.cat([torch.zeros(3, dtype=torch.float32, device=device), jaw]).view(1, -1)

    code = {"exp": exp, "pose": pose}
    for key, value in default_code.items():
        if key in {"exp", "pose"}:
            continue
        if torch.is_tensor(value):
            code[key] = value.detach().clone().float().to(device)
        else:
            code[key] = value
    if fix_cam and "cam" in code:
        # Match the camera convention used by the official EmoAva visualize.py.
        code["cam"][0, 0] = 5.0
        code["cam"][0, 1] = 0.0
        code["cam"][0, 2] = 0.05
    return code


def output_name(row: int, records: list[dict] | None) -> str:
    if records is None or row >= len(records):
        return f"row_{row:02d}.mp4"
    case_id = records[row].get("case_id_1based", row + 1)
    return f"row_{row:02d}_case_{int(case_id):04d}.mp4"


def render_sequence(deca, default_code, seq: np.ndarray, device: torch.device, args) -> list[np.ndarray]:
    frames = []
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 53:
        raise ValueError(f"Expected one expression sequence with shape [T, 53], got {seq.shape}.")

    if args.max_frames > 0 and len(seq) > args.max_frames:
        idx = np.linspace(0, len(seq) - 1, args.max_frames).astype(np.int64)
        seq = seq[idx]

    with torch.no_grad():
        for frame in tqdm(seq, leave=False):
            code = make_code(default_code, frame, device=device, fix_cam=not args.no_fix_cam)
            if args.no_detail:
                if args.vis_key != "shape_images":
                    raise ValueError("--no-detail only supports --vis-key shape_images.")
                opdict = deca.decode(code, rendering=False, return_vis=False, use_detail=False, vis_lmk=False)
                shape_images = deca.render.render_shape(opdict["verts"], opdict["trans_verts"])
                image = deca.visualize({"shape_images": shape_images}, size=args.size)
            else:
                _, visdict = deca.decode(code, rendering=True, return_vis=True, use_detail=True)
                if args.vis_key not in visdict:
                    available = ", ".join(visdict.keys())
                    raise KeyError(f"Visualization key `{args.vis_key}` is unavailable. Available keys: {available}")
                image = deca.visualize({args.vis_key: visdict[args.vis_key]}, size=args.size)
            frames.append(image)
    return frames


def write_video(frames: list[np.ndarray], path: Path, fps: float):
    if not frames:
        raise ValueError("No frames were rendered.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_size = (frames[0].shape[1], frames[0].shape[0])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for frame in frames:
        # Keep the same color conversion convention as official visualize.py.
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render EmoAva 53-D expression sequences with the official DECA backend.")
    parser.add_argument("--exp-path", required=True, help="Pickle list of [T,53] expression sequences.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--emoava-src", default="external/EmoAva/src")
    parser.add_argument("--default-code", default="external/EmoAva/src/data/default_code_trevor_emoca2.pkl")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--geometry-only", action="store_true", help="Disable FLAME texture space; renders official geometry.")
    parser.add_argument("--rasterizer", default="pytorch3d", choices=["pytorch3d", "standard"])
    parser.add_argument("--vis-key", default="shape_images", choices=["shape_images", "shape_detail_images", "rendered_images"])
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-detail", action="store_true", help="Use only coarse DECA decode. Usually leave this off.")
    parser.add_argument("--no-fix-cam", action="store_true", help="Do not apply the official fixed camera values.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    repo = Path(__file__).resolve().parents[1]
    exp_path = (repo / args.exp_path).resolve()
    output_dir = (repo / args.output_dir).resolve()
    emoava_src = (repo / args.emoava_src).resolve()
    default_code_path = (repo / args.default_code).resolve()

    device = torch.device(args.device)
    DECA, get_cfg_defaults = import_official_deca(emoava_src)
    cfg = get_cfg_defaults()
    cfg.rasterizer_type = args.rasterizer
    if args.geometry_only:
        cfg.model.use_tex = False
        if args.vis_key == "rendered_images":
            raise ValueError("`rendered_images` requires texture mode; remove --geometry-only.")

    exps = load_pickle(exp_path, map_location=args.device)
    default_code = load_pickle(default_code_path, map_location=args.device)
    deca = DECA(config=cfg, device=args.device)
    deca.eval()

    records = load_manifest_records(exp_path)
    total = len(exps) if args.max_cases <= 0 else min(args.max_cases, len(exps))
    for row in tqdm(range(total), desc="render cases"):
        frames = render_sequence(deca, default_code, exps[row], device=device, args=args)
        write_video(frames, output_dir / output_name(row, records), fps=args.fps)

    print(output_dir)


if __name__ == "__main__":
    main()
