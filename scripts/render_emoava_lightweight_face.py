import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch


def normalize(x: float, scale: float = 2.5) -> float:
    return float(np.tanh(x / scale))


def controls(frame: np.ndarray) -> dict[str, float]:
    # The official 53-D vector is FLAME expression + jaw-like pose. This renderer
    # is only a lightweight qualitative proxy, not a FLAME renderer.
    mouth = normalize(np.mean(frame[0:10]))
    brow = normalize(np.mean(frame[10:20]))
    eye = normalize(np.mean(frame[20:30]))
    cheek = normalize(np.mean(frame[30:40]))
    jaw = normalize(np.mean(frame[50:53]), 1.5)
    energy = normalize(np.linalg.norm(frame), 8.0)
    return {
        "mouth": mouth,
        "brow": brow,
        "eye": eye,
        "cheek": cheek,
        "jaw": jaw,
        "energy": energy,
    }


def draw_frame(frame: np.ndarray, title: str, out_path: Path):
    c = controls(frame)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    face = plt.Circle((0, 0), 1.0, color="#f2d0b7", ec="#3b302c", lw=2)
    ax.add_patch(face)

    brow_y = 0.43 + 0.12 * c["brow"]
    brow_tilt = 0.08 * c["brow"]
    ax.plot([-0.55, -0.20], [brow_y + brow_tilt, brow_y - brow_tilt], color="#2d2623", lw=4)
    ax.plot([0.20, 0.55], [brow_y - brow_tilt, brow_y + brow_tilt], color="#2d2623", lw=4)

    eye_h = max(0.025, 0.08 + 0.06 * c["eye"])
    for x in [-0.38, 0.38]:
        eye = Ellipse((x, 0.24), 0.24, eye_h, color="#1e1b1a")
        ax.add_patch(eye)

    cheek_alpha = min(max(0.12 + 0.25 * c["cheek"], 0.05), 0.35)
    for x in [-0.48, 0.48]:
        cheek = plt.Circle((x, -0.05), 0.16, color="#d96b6b", alpha=cheek_alpha)
        ax.add_patch(cheek)

    jaw_drop = 0.16 + 0.20 * max(c["jaw"], 0)
    smile = 0.22 * c["mouth"]
    mouth_x = np.linspace(-0.42, 0.42, 60)
    mouth_y = -0.43 - jaw_drop * 0.2 + smile * (1 - (mouth_x / 0.42) ** 2)
    if c["jaw"] > 0.25:
        mouth = Ellipse((0, -0.43), 0.55, 0.18 + 0.25 * c["jaw"], color="#5a2020")
        ax.add_patch(mouth)
    else:
        ax.plot(mouth_x, mouth_y, color="#5a2020", lw=5, solid_capstyle="round")

    ax.text(0, 1.13, title, ha="center", va="center", fontsize=9)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default="outputs/when_words_smile_repro/test_parallel_full.pt")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--output-dir", default="outputs/when_words_smile_repro/light_face")
    args = parser.parse_args()

    pred = torch.load(args.prediction, map_location="cpu").numpy()
    seq = pred[args.case]
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / f"case_{args.case + 1:02d}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(seq) - 1, args.frames).astype(int)

    images = []
    for out_i, src_i in enumerate(idx):
        path = frame_dir / f"frame_{out_i:03d}.png"
        draw_frame(seq[src_i], f"case {args.case + 1} frame {src_i}", path)
        images.append(imageio.imread(path))

    gif_path = output_dir / f"case_{args.case + 1:02d}.gif"
    imageio.mimsave(gif_path, images, duration=0.08)
    print(gif_path)


if __name__ == "__main__":
    main()
