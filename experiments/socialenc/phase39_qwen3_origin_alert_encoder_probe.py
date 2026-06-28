"""Phase39: Qwen3-Embedding-4B probe for Experiment 3 origin alert.

This uses the main paper task from phase28:
  - pre-popularity origin alert
  - origin window first10
  - semantic threshold 0.55 only
  - encoder input is origin tweet text only

It is separate from phase28 so Qwen3 can be evaluated before being added to the
main text-encoder roster.
"""
from __future__ import annotations

import pathlib

import numpy as np

import phase21_streaming_agent_encoder_baselines as p21
import phase28_origin_alert_encoder_baselines as p28


QWEN3_4B_SLUG = "qwen3_embedding_4b_st"
QWEN3_4B_CONFIG = {
    "model": "Qwen/Qwen3-Embedding-4B",
    "prefix": "",
    "pooling": "sentence_transformer",
}


def encode_qwen3_st_texts(slug: str, config: dict, texts: list[str]) -> dict[str, np.ndarray]:
    cache = p21.load_embedding_cache(slug)
    missing = [text for text in texts if text not in cache]
    p21.log(f"  {slug}: cached={len(cache)} missing={len(missing)}")
    if not missing:
        return cache

    import torch
    from sentence_transformers import SentenceTransformer

    kwargs = {"model_kwargs": {"dtype": torch.float16}} if torch.cuda.is_available() else {}
    model = SentenceTransformer(config["model"], **kwargs)
    for start in range(0, len(missing), p21.BATCH_SIZE):
        batch = [config["prefix"] + text for text in missing[start:start + p21.BATCH_SIZE]]
        embeddings = model.encode(
            batch,
            batch_size=p21.BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for text, vector in zip(missing[start:start + p21.BATCH_SIZE], embeddings):
            cache[text] = vector.astype(np.float32)
        done = min(start + p21.BATCH_SIZE, len(missing))
        if done % (p21.BATCH_SIZE * 10) == 0 or done == len(missing):
            p21.log(f"  {slug}: encoded {done}/{len(missing)}")

    p21.save_embedding_cache(slug, cache)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def main() -> None:
    p21.BATCH_SIZE = 4
    p21.MODEL_CONFIGS = {QWEN3_4B_SLUG: QWEN3_4B_CONFIG}
    p21.encode_texts = encode_qwen3_st_texts
    p28.THRESHOLDS = [0.55]
    p28.ENCODER_SLUGS = [QWEN3_4B_SLUG]
    p28.OUT = (
        pathlib.Path(__file__).with_name("phase39_qwen3_origin_alert_encoder_probe_result.json")
    )
    p28.main()


if __name__ == "__main__":
    main()
