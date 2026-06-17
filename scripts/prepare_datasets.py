import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[skip] invalid json at {path}:{line_no}: {exc}")


def split_text(text: str, chunk_size: int):
    text = text.strip()
    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            yield chunk


def prepare_pretrain(input_path: Path, output_path: Path, chunk_size: int, max_records: int | None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for item in iter_jsonl(input_path):
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            seen += 1
            for chunk in split_text(text, chunk_size):
                out.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                written += 1

            if max_records is not None and seen >= max_records:
                break

    print(f"Pretrain records read: {seen}")
    print(f"Pretrain chunks written: {written}")
    print(f"Pretrain output: {output_path}")


def convert_message(conversations):
    messages = [{"role": "system", "content": "你是一个AI助手"}]

    for item in conversations:
        role = item.get("from")
        content = item.get("value")
        if not isinstance(content, str) or not content.strip():
            continue

        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

    return messages


def prepare_sft(input_path: Path, output_path: Path, max_records: int | None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for item in iter_jsonl(input_path):
            conversations = item.get("conversations")
            if not isinstance(conversations, list):
                continue

            seen += 1
            messages = convert_message(conversations)
            if any(message["role"] == "assistant" for message in messages):
                out.write(json.dumps(messages, ensure_ascii=False) + "\n")
                written += 1

            if max_records is not None and seen >= max_records:
                break

    print(f"SFT records read: {seen}")
    print(f"SFT conversations written: {written}")
    print(f"SFT output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Happy-LLM pretrain and SFT datasets.")
    parser.add_argument("--dataset_dir", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--seq_input", type=Path, default=None)
    parser.add_argument("--belle_input", type=Path, default=None)
    parser.add_argument("--pretrain_output", type=Path, default=None)
    parser.add_argument("--sft_output", type=Path, default=None)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_pretrain_records", type=int, default=None)
    parser.add_argument("--max_sft_records", type=int, default=None)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--skip_sft", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    seq_input = args.seq_input or dataset_dir / "mobvoi_seq_monkey_general_open_corpus.jsonl"
    belle_input = args.belle_input or dataset_dir / "BelleGroup" / "train_3.5M_CN.json"
    pretrain_output = args.pretrain_output or dataset_dir / "seq_monkey_datawhale.jsonl"
    sft_output = args.sft_output or dataset_dir / "BelleGroup_sft.jsonl"

    if not args.skip_pretrain:
        if not seq_input.exists():
            raise FileNotFoundError(f"Seq-monkey source file not found: {seq_input}")
        prepare_pretrain(seq_input, pretrain_output, args.chunk_size, args.max_pretrain_records)

    if not args.skip_sft:
        if not belle_input.exists():
            raise FileNotFoundError(f"BelleGroup source file not found: {belle_input}")
        prepare_sft(belle_input, sft_output, args.max_sft_records)


if __name__ == "__main__":
    main()
