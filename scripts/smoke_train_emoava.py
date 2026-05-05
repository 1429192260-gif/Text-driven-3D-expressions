import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", default="models/bert-base-cased")
    parser.add_argument("--output", default="outputs/when_words_smile_repro/train_smoke/model_smoke.chkpt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--trg-len", type=int, default=64)
    parser.add_argument("--src-len", type=int, default=64)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "external" / "EmoAva" / "src"))
    from em_dataloader import get_dataloader, load_dataset_split_batch
    from train import cal_loss_continue
    from transformer.Models import Transformer
    from transformer.Optim import ScheduledOptim
    from transformers import BertModel, BertTokenizer

    device = torch.device("cpu")
    src_texts, tgt_exps, tgt_steps = load_dataset_split_batch("train", 4, None)
    tokenizer = BertTokenizer.from_pretrained(repo / args.pretrained_path)
    tgt_texts = [" ".join(["[UNK]"] * i) for i in tgt_steps]
    loader = get_dataloader(
        src_texts,
        tgt_texts,
        tgt_exps,
        {
            "tokenizer": tokenizer,
            "batch_size": args.batch_size,
            "src_len": args.src_len,
            "trg_len": args.trg_len,
            "trg_exp_dim": 53,
            "shuffle": False,
        },
    )
    encoder = BertModel.from_pretrained(repo / args.pretrained_path)
    model = Transformer(
        tokenizer.vocab_size,
        53,
        src_pad_idx=0,
        trg_pad_idx=0,
        d_k=64,
        d_v=64,
        d_model=768,
        d_word_vec=768,
        d_inner=2048,
        n_layers=1,
        n_layers_e=12,
        n_head=12,
        dropout=0.1,
        scale_emb=True,
        cvae=True,
        flat_method="casual_attention",
        feature_attention=True,
        encoder_pretrained=encoder,
        one_step_loss=True,
        exp_only_dim=50,
        src_len=args.src_len,
        trg_len=args.trg_len,
        fix_encoder=True,
        second_sample="one",
    ).to(device)
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
        2.0,
        768,
        4000,
    )
    batch = next(iter(loader))
    src_seq = batch["src_input_ids"].to(device)
    src_mask = batch["src_attention_mask"].to(device)
    tgt_seq = batch["tgt_input_ids"].to(device)
    tgt_mask = batch["tgt_attention_mask"].to(device)
    tgt_exp_mask = batch["tgt_exp_mask"].to(device)
    tgt_exp = batch["tgt_exp"].to(device)
    gold = batch["tgt_exp_gold"].to(device)
    gold_mask = batch["tgt_gold_mask"].to(device)

    model.train()
    pred, kl_loss, pred_pri_z = model(src_seq, src_mask, tgt_seq, tgt_mask, tgt_exp, tgt_exp_mask, "train")
    loss, first_token_loss = cal_loss_continue(pred, gold, gold_mask, True)
    one_step_loss = cal_loss_continue(pred_pri_z, gold[:, :-1, :], gold_mask[:, :-1])
    total_loss = loss + kl_loss + one_step_loss
    total_loss.backward()
    optimizer.step_and_update_lr()

    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "loss": float(loss.item()),
            "kl_loss": float(kl_loss.item()),
            "one_step_loss": float(one_step_loss.item()),
            "first_token_loss": float(first_token_loss.item()),
        },
        out,
    )
    print(
        {
            "loss": float(loss.item()),
            "kl_loss": float(kl_loss.item()),
            "one_step_loss": float(one_step_loss.item()),
            "first_token_loss": float(first_token_loss.item()),
            "output": str(out),
        }
    )


if __name__ == "__main__":
    main()
