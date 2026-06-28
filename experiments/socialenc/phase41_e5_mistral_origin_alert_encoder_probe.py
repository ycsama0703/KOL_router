"""Phase41: E5-Mistral-7B-Instruct probe for Experiment 3 origin alert.

Same task as phase28:
  - pre-popularity origin alert
  - origin window first10
  - semantic threshold 0.55 only
  - encoder input is origin tweet text only

This uses SentenceTransformer for intfloat/e5-mistral-7b-instruct. The model is
large enough that luyao4's 16GB GPU should use batch size 1.
"""
from __future__ import annotations

import pathlib

import numpy as np

import phase21_streaming_agent_encoder_baselines as p21
import phase28_origin_alert_encoder_baselines as p28


E5_MISTRAL_SLUG = "e5_mistral_7b_instruct"
E5_MISTRAL_CONFIG = {
    "model": "intfloat/e5-mistral-7b-instruct",
    "prefix": "",
    "pooling": "sentence_transformer",
}


def encode_e5_mistral_texts(slug: str, config: dict, texts: list[str]) -> dict[str, np.ndarray]:
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
        if done % (p21.BATCH_SIZE * 50) == 0 or done == len(missing):
            p21.log(f"  {slug}: encoded {done}/{len(missing)}")

    p21.save_embedding_cache(slug, cache)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def main() -> None:
    p21.BATCH_SIZE = 1
    p21.MODEL_CONFIGS = {E5_MISTRAL_SLUG: E5_MISTRAL_CONFIG}
    p21.encode_texts = encode_e5_mistral_texts
    p28.THRESHOLDS = [0.55]
    p28.ENCODER_SLUGS = [E5_MISTRAL_SLUG]
    p28.OUT = (
        pathlib.Path(__file__).with_name("phase41_e5_mistral_origin_alert_encoder_probe_result.json")
    )
    p28.main()


if __name__ == "__main__":
    main()
