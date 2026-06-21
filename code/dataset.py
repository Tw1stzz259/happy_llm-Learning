import json

import torch
from torch.utils.data import Dataset


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._offsets = []

        with open(data_path, "rb") as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break
                self._offsets.append(offset)

    def __len__(self):
        return len(self._offsets)

    def _read_item(self, index):
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            return json.loads(f.readline().decode("utf-8"))

    def _build_tensors(self, input_ids, loss_mask):
        padding_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.padding] * padding_len
        loss_mask = loss_mask + [0] * padding_len

        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(input_ids[1:], dtype=torch.long)
        shifted_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        y[shifted_mask == 0] = -100
        return x, y, shifted_mask


class PretrainDataset(JsonlDataset):
    def __getitem__(self, index):
        sample = self._read_item(index)
        content_ids = self.tokenizer(
            sample["text"],
            add_special_tokens=False,
        ).input_ids

        input_ids = [self.tokenizer.bos_token_id]
        input_ids += content_ids[: self.max_length - 2]
        input_ids += [self.tokenizer.eos_token_id]
        loss_mask = [1] * len(input_ids)
        return self._build_tensors(input_ids, loss_mask)


class SFTDataset(JsonlDataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__(data_path, tokenizer, max_length)
        self.assistant_marker = tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        ).input_ids

    def _marker_positions(self, input_ids):
        marker_length = len(self.assistant_marker)
        return [
            i
            for i in range(len(input_ids) - marker_length + 1)
            if input_ids[i : i + marker_length] == self.assistant_marker
        ]

    def _truncate_input_ids(self, input_ids):
        if len(input_ids) <= self.max_length:
            return input_ids

        marker_positions = self._marker_positions(input_ids)
        if not marker_positions:
            return input_ids[: self.max_length]

        last_marker = marker_positions[-1]
        window_start = max(0, len(input_ids) - self.max_length)
        if window_start > last_marker:
            window_start = last_marker
        return input_ids[window_start : window_start + self.max_length]

    def generate_loss_mask(self, input_ids):
        mask = [0] * len(input_ids)
        marker_length = len(self.assistant_marker)
        i = 0

        while i <= len(input_ids) - marker_length:
            if input_ids[i : i + marker_length] != self.assistant_marker:
                i += 1
                continue

            start = i + marker_length
            end = len(input_ids) - 1
            for position in range(start, len(input_ids)):
                if input_ids[position] == self.tokenizer.eos_token_id:
                    end = position
                    break

            if start <= end:
                for position in range(start, end + 1):
                    mask[position] = 1
            i = end + 1

        return mask

    def __getitem__(self, index):
        sample = self._read_item(index)
        text = self.tokenizer.apply_chat_template(
            sample,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids
        input_ids = self._truncate_input_ids(input_ids)
        loss_mask = self.generate_loss_mask(input_ids)
        return self._build_tensors(input_ids, loss_mask)
