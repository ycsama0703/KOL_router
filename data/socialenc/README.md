# Socialenc Data Panel

This directory contains the derived input data used by the retained KOL router
experiments.

Files:

```text
{SYMBOL}.jsonl   tweet/KOL metadata and text
{SYMBOL}.npz     MiniLM embeddings aligned to JSONL row order
SHA256SUMS.txt   checksums
file_sizes.csv   byte sizes
```

The scripts read this directory through:

```python
pathlib.Path(__file__).resolve().parents[2] / "data/socialenc"
```

The broader internal project also contained `windows.npz`, but that file is not
included because the retained paper experiments do not read it.
