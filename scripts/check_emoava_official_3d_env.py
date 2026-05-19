import argparse
import importlib.util
import io
import pickle
import sys
from pathlib import Path

import torch


def check_module(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, "missing"
    return True, str(spec.origin)


def patch_torch_pickle_load(map_location: str):
    def _load_from_bytes(buffer):
        return torch.load(io.BytesIO(buffer), map_location=map_location)

    torch.storage._load_from_bytes = _load_from_bytes


def load_default_code(path: Path, map_location: str):
    patch_torch_pickle_load(map_location)
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Check whether the official EmoAva DECA/FLAME 3D backend can run.")
    parser.add_argument("--emoava-src", default="external/EmoAva/src")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--instantiate-deca", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    emoava_src = (repo / args.emoava_src).resolve()
    data_dir = emoava_src / "data"
    default_code = data_dir / "default_code_trevor_emoca2.pkl"
    required_files = [
        data_dir / "generic_model.pkl",
        data_dir / "head_template.obj",
        data_dir / "landmark_embedding.npy",
        data_dir / "mean_texture.jpg",
        data_dir / "uv_face_eye_mask.png",
        data_dir / "uv_face_mask.png",
        default_code,
    ]
    optional_files = [
        data_dir / "deca_model.tar",
        data_dir / "FLAME_albedo_from_BFM.npz",
    ]

    print("== Torch ==")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"torch cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    print("\n== Python Modules ==")
    for name in ["cv2", "yacs", "kornia", "skimage", "tqdm", "pytorch3d"]:
        ok, detail = check_module(name)
        print(f"{name}: {'ok' if ok else 'missing'} ({detail})")

    print("\n== Official Files ==")
    for path in required_files:
        print(f"{path.relative_to(repo)}: {'ok' if path.exists() else 'missing'}")
    for path in optional_files:
        print(f"{path.relative_to(repo)}: {'ok' if path.exists() else 'optional missing'}")

    print("\n== Default Code ==")
    try:
        code = load_default_code(default_code, map_location=args.device if torch.cuda.is_available() else "cpu")
        print("default_code: ok")
        print({k: tuple(v.shape) for k, v in code.items() if hasattr(v, "shape")})
    except Exception as exc:
        print(f"default_code: failed ({exc})")

    if args.instantiate_deca:
        print("\n== DECA Instantiate ==")
        try:
            sys.path.insert(0, str(emoava_src))
            from decalib.deca import DECA
            from decalib.utils.config import get_cfg_defaults

            cfg = get_cfg_defaults()
            cfg.model.use_tex = False
            DECA(config=cfg, device=args.device)
            print("DECA geometry-only init: ok")
        except Exception as exc:
            print(f"DECA geometry-only init: failed ({exc})")


if __name__ == "__main__":
    main()
