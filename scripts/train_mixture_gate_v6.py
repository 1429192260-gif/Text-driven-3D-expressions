import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def pad_gold(seq: np.ndarray, max_len: int, dim: int = 53) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_len, dim), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=bool)
    usable = min(len(seq), max_len)
    if usable:
        out[:usable] = seq[:usable]
        mask[:usable] = True
    return out, mask


def load_split(prediction: Path, dataset_dir: Path, split: str):
    pred = torch.load(prediction, map_location="cpu").float()
    texts = load_pickle(dataset_dir / f"{split}_stage1_text.pkl")
    gold_raw = load_pickle(dataset_dir / f"{split}_stage1_exps.pkl")
    gold = []
    mask = []
    for seq in gold_raw[: pred.shape[0]]:
        g, m = pad_gold(np.asarray(seq, dtype=np.float32), pred.shape[1], pred.shape[2])
        gold.append(g)
        mask.append(m)
    return pred, torch.tensor(np.stack(gold)), torch.tensor(np.stack(mask)), texts[: pred.shape[0]]


def masked_l1(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = torch.abs(pred - gold).mean(dim=-1)
    return err[mask].mean()


def masked_temporal_l1(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    diff_pred = pred[:, 1:] - pred[:, :-1]
    diff_gold = gold[:, 1:] - gold[:, :-1]
    diff_mask = mask[:, 1:] & mask[:, :-1]
    err = torch.abs(diff_pred - diff_gold).mean(dim=-1)
    return err[diff_mask].mean()


class MixtureGate(nn.Module):
    def __init__(self, exp_dim: int = 53, hidden: int = 64, mode: str = "frame"):
        super().__init__()
        self.mode = mode
        input_dim = exp_dim * 4 + 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, base: torch.Tensor, global_pred: torch.Tensor, prior_pred: torch.Tensor):
        diff_global = global_pred - base
        diff_prior = prior_pred - base
        disagreement = prior_pred - global_pred
        energy = torch.stack(
            [
                torch.linalg.norm(base, dim=-1),
                torch.linalg.norm(diff_global, dim=-1),
                torch.linalg.norm(diff_prior, dim=-1),
                torch.linalg.norm(disagreement, dim=-1),
            ],
            dim=-1,
        )
        frame_features = torch.cat([base, diff_global, diff_prior, disagreement, energy], dim=-1)
        if self.mode == "sample":
            pooled = frame_features.mean(dim=1)
            gate = torch.sigmoid(self.net(pooled))[:, None, :]
        else:
            gate = torch.sigmoid(self.net(frame_features))
        mixed = global_pred + gate * (prior_pred - global_pred)
        return mixed, gate


def evaluate(model, base, global_pred, prior_pred, gold, mask, batch_size, device):
    model.eval()
    losses = []
    outputs = []
    gates = []
    with torch.no_grad():
        for start in range(0, base.shape[0], batch_size):
            end = min(start + batch_size, base.shape[0])
            b = base[start:end].to(device)
            g = global_pred[start:end].to(device)
            p = prior_pred[start:end].to(device)
            y = gold[start:end].to(device)
            m = mask[start:end].to(device)
            mixed, gate = model(b, g, p)
            losses.append(masked_l1(mixed, y, m).detach().cpu())
            outputs.append(mixed.detach().cpu())
            gates.append(gate.detach().cpu())
    return float(torch.stack(losses).mean().item()), torch.cat(outputs, dim=0), torch.cat(gates, dim=0)


def train_epoch(
    model,
    optimizer,
    base,
    global_pred,
    prior_pred,
    gold,
    mask,
    indices,
    batch_size,
    device,
    dynamic_weight,
    gate_reg_weight,
):
    model.train()
    order = indices[torch.randperm(len(indices))]
    total_loss = 0.0
    total_batches = 0
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        b = base[idx].to(device)
        g = global_pred[idx].to(device)
        p = prior_pred[idx].to(device)
        y = gold[idx].to(device)
        m = mask[idx].to(device)
        mixed, gate = model(b, g, p)
        reconstruction = masked_l1(mixed, y, m)
        dynamic = masked_temporal_l1(mixed, y, m)
        gate_reg = gate_reg_weight * (gate * (1.0 - gate)).mean()
        loss = reconstruction + dynamic_weight * dynamic + gate_reg
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def save_report(path: Path, args, history, best_val, test_gates):
    lines = [
        "# When Words Smile Prior V6 Training Report",
        "",
        "## Method",
        "",
        "V6 trains a mixture gate between the strongest global residual correction output and the learned-prior output. The gate learns when to trust the learned affect-prior branch.",
        "",
        "```text",
        "P_v6 = P_global + gamma * (P_prior - P_global)",
        "```",
        "",
        "The gate is trained only on the dev split and then evaluated on the test split.",
        "",
        "## Settings",
        "",
        f"- Gate mode: `{args.mode}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.lr}`",
        f"- Hidden size: `{args.hidden}`",
        f"- Dynamic loss weight: `{args.dynamic_weight}`",
        f"- Gate regularization weight: `{args.gate_reg_weight}`",
        f"- Best val MAE: `{best_val:.6f}`",
        f"- Output: `{args.output}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Test Gate Statistics",
        "",
        f"- Mean gate: `{float(test_gates.mean()):.6f}`",
        f"- Median gate: `{float(test_gates.median()):.6f}`",
        f"- Min gate: `{float(test_gates.min()):.6f}`",
        f"- Max gate: `{float(test_gates.max()):.6f}`",
        "",
        "## History",
        "",
        "| Epoch | Train Loss | Val MAE |",
        "|---:|---:|---:|",
    ]
    for epoch, train_loss, val_loss in history:
        lines.append(f"| {epoch} | {train_loss:.6f} | {val_loss:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-base", default="outputs/when_words_smile_prior_v1/dev_parallel_full.pt")
    parser.add_argument("--test-base", default="outputs/when_words_smile_repro/test_parallel_full.pt")
    parser.add_argument("--dev-global", required=True)
    parser.add_argument("--test-global", default="outputs/when_words_smile_prior_v4/test_uncertainty_prior_fusion_no_prior_branch.pt")
    parser.add_argument("--dev-prior", required=True)
    parser.add_argument("--test-prior", default="outputs/when_words_smile_prior_v5/test_learned_affect_prior_fusion.pt")
    parser.add_argument("--dataset-dir", default="external/EmoAva/dataset")
    parser.add_argument("--output", default="outputs/when_words_smile_prior_v6/test_mixture_gate.pt")
    parser.add_argument("--checkpoint", default="outputs/when_words_smile_prior_v6/mixture_gate_v6.pt")
    parser.add_argument("--report", default="docs/when_words_smile_prior_v6_train_report.md")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dynamic-weight", type=float, default=0.05)
    parser.add_argument("--gate-reg-weight", type=float, default=0.0002)
    parser.add_argument("--mode", choices=["frame", "sample"], default="frame")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / args.dataset_dir

    dev_base, dev_gold, dev_mask, _ = load_split(repo / args.dev_base, dataset_dir, "dev")
    test_base, test_gold, test_mask, _ = load_split(repo / args.test_base, dataset_dir, "test")
    dev_global = torch.load(repo / args.dev_global, map_location="cpu").float()
    test_global = torch.load(repo / args.test_global, map_location="cpu").float()
    dev_prior = torch.load(repo / args.dev_prior, map_location="cpu").float()
    test_prior = torch.load(repo / args.test_prior, map_location="cpu").float()

    train_size = min(args.train_size, dev_base.shape[0] - 1)
    train_idx = torch.arange(train_size)
    val_idx = torch.arange(train_size, dev_base.shape[0])
    model = MixtureGate(exp_dim=dev_base.shape[-1], hidden=args.hidden, mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            optimizer,
            dev_base,
            dev_global,
            dev_prior,
            dev_gold,
            dev_mask,
            train_idx,
            args.batch_size,
            device,
            args.dynamic_weight,
            args.gate_reg_weight,
        )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            val_loss, _, _ = evaluate(
                model,
                dev_base[val_idx],
                dev_global[val_idx],
                dev_prior[val_idx],
                dev_gold[val_idx],
                dev_mask[val_idx],
                args.batch_size,
                device,
            )
            history.append((epoch, train_loss, val_loss))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_mae={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_output, test_gates = evaluate(
        model,
        test_base,
        test_global,
        test_prior,
        test_gold,
        test_mask,
        args.batch_size,
        device,
    )
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_output, output)
    checkpoint = repo / args.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "best_val_mae": best_val, "history": history, "args": vars(args)}, checkpoint)
    save_report(repo / args.report, args, history, best_val, test_gates)
    print(repo / args.report)
    print(output)


if __name__ == "__main__":
    main()
