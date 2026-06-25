import argparse
import os
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer

from config import ModelConfig
from model import Transformer


def resolve_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def strip_checkpoint_prefixes(state_dict):
    for prefix in ("_orig_mod.", "module."):
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = state_dict.pop(key)
    return state_dict


def load_model(args, tokenizer):
    config = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
    )
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id

    model = Transformer(config)
    state_dict = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(strip_checkpoint_prefixes(state_dict), strict=True)
    model.to(args.device)
    model.eval()
    return model


def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def reply(model, tokenizer, messages, args):
    prompt = build_prompt(tokenizer, messages)
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(input_ids) > args.max_seq_len:
        input_ids = input_ids[-args.max_seq_len :]

    x = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    eos_id = tokenizer.eos_token_id

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)

    with ctx:
        output_ids = model.generate(
            x,
            stop_id=eos_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0].tolist()

    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    text = text.split(tokenizer.eos_token, 1)[0]
    return text.strip()


def chat_loop(model, tokenizer, args):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        answer = reply(model, tokenizer, messages, args)
        print(answer)
        return

    print("Enter empty line, /exit, or /quit to stop.")
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in {"/exit", "/quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        answer = reply(model, tokenizer, messages, args)
        print(f"Assistant: {answer}")
        messages.append({"role": "assistant", "content": answer})


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the Tiny-LLM SFT checkpoint.")
    parser.add_argument("--ckpt", type=str, default="./sft_model_40M/sft_576_9_6144_final.pth")
    parser.add_argument("--tokenizer_path", type=str, default="./Tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])

    parser.add_argument("--dim", type=int, default=576)
    parser.add_argument("--n_layers", type=int, default=9)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--system", type=str, default="你是一个有帮助的中文助手。")
    parser.add_argument("--prompt", type=str, default=None)

    args = parser.parse_args()
    args.ckpt = resolve_path(args.ckpt)
    args.tokenizer_path = resolve_path(args.tokenizer_path)
    return args


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args, tokenizer)
    chat_loop(model, tokenizer, args)
