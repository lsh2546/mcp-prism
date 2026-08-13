from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


@dataclass(frozen=True)
class EncoderConfig:
    model_path: Path
    tokenizer_path: Path
    max_length: int = 128
    threads: int = 2


class Int8OnnxEncoder:
    """Pinned INT8 sentence encoder. Missing/non-INT8 models fail closed."""

    def __init__(self, config: EncoderConfig):
        if not config.model_path.is_file() or not config.tokenizer_path.is_file():
            raise FileNotFoundError("INT8 model and tokenizer are required; run scripts/fetch_model.py")
        if "int8" not in config.model_path.name.lower() and "qint8" not in config.model_path.name.lower():
            raise ValueError("retrieval model filename must identify the pinned INT8 artifact")
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(config.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=config.max_length)
        self.tokenizer.enable_padding(length=None)
        self.input_names = {node.name for node in self.session.get_inputs()}

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, 384), dtype=np.float32)
        encoded = self.tokenizer.encode_batch(values)
        ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        masks = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": masks}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        output = self.session.run(None, feeds)[0]
        expanded = masks[..., None].astype(np.float32)
        pooled = (output * expanded).sum(axis=1) / np.clip(expanded.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)

