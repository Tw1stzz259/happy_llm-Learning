import os
import argparse
import time
import math
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from config import ModelConfig
from model import Transformer
from dataset import SFTDataset


def Logger(context):
    print(context)


def resolve_path(path):
    if path and path.lower() in {"none", "null"}:
        return path.lower()
    if not path or os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def get_lr(it, all_iters):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all_iters
    min_lr = args.learning_rate / 10

    if warmup_iters > 0 and it < warmup_iters:
        return args.learning_rate * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_iters = max(lr_decay_iters - warmup_iters, 1)
    decay_ratio = (it - warmup_iters) / decay_iters
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch):
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        valid_tokens = loss_mask.sum()
        if valid_tokens.item() == 0:
            continue

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / valid_tokens

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_time:{}min".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f"{args.save_dir}/sft_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth"
            torch.save(model.state_dict(), ckp)
            model.train()

        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f"{args.save_dir}/sft_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step + 1}.pth"
            torch.save(model.state_dict(), ckp)
            model.train()


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path or checkpoint_path.lower() in {"none", "null"}:
        Logger("No pretrain checkpoint provided; SFT will start from random weights.")
        return

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=args.device)
    for prefix in ("_orig_mod.", "module."):
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = state_dict.pop(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    Logger(f"Loaded pretrain checkpoint: {checkpoint_path}")
    if missing:
        Logger(f"Missing keys: {len(missing)}")
    if unexpected:
        Logger(f"Unexpected keys: {len(unexpected)}")


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is not None:
        lm_config.pad_token_id = tokenizer.pad_token_id

    model = Transformer(lm_config)
    load_checkpoint(model, args.pretrain_ckpt)
    model = model.to(args.device)

    Logger(f"LLM total parameters: {count_parameters(model) / 1e6:.3f}M")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-LLM SFT")

    parser.add_argument("--out_dir", type=str, default="sft_model_40M", help="model output directory")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="mixed precision dtype")
    parser.add_argument("--dim", type=int, default=576, help="model dimension")
    parser.add_argument("--n_layers", type=int, default=9, help="Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=8, help="KV attention heads")
    parser.add_argument("--max_seq_len", type=int, default=512, help="max sequence length")

    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft.jsonl", help="SFT jsonl data path")
    parser.add_argument("--tokenizer_path", type=str, default="./Tokenizer", help="tokenizer path")
    parser.add_argument("--pretrain_ckpt", type=str, default="./base_model_40M/pretrain_576_9_6144_final.pth", help="pretrain checkpoint path")

    parser.add_argument("--accumulation_steps", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--warmup_iters", type=int, default=0, help="warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="checkpoint save interval")

    args = parser.parse_args()

    lm_config = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
    )
    max_seq_len = lm_config.max_seq_len

    args.save_dir = resolve_path(args.out_dir)
    args.data_path = resolve_path(args.data_path)
    args.tokenizer_path = resolve_path(args.tokenizer_path)
    args.pretrain_ckpt = resolve_path(args.pretrain_ckpt)
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(42)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda", dtype=ptdtype)

    model, tokenizer = init_model()

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=(device_type == "cuda"),
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)

    final_ckp = f"{args.save_dir}/sft_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_final.pth"
    torch.save(model.state_dict(), final_ckp)
    Logger(f"Final model saved to {final_ckp}")
