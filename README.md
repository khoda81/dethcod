# dethcod

Research experiments on **learned lossless compression with transformer bottlenecks**.

The core question is whether a learned compressor can transform an input into a shorter intermediate representation that a language-model-style decoder can use to reconstruct the original data exactly, while optimizing the trade-off between representation cost and reconstruction/modeling cost.

> **Repository status:** this public tree is an early research snapshot. It contains exploratory notebooks and prototype model code rather than a polished production codec or a fully reproducible benchmark suite.

## Experiments in this snapshot

### Token bottleneck

`dethcod/token/` explores a T5-based sequence-to-sequence compressor. The compression model extends `T5ForConditionalGeneration` and includes a critic/value head for experiments where generation decisions are optimized with a learned objective.

The accompanying `TokenDethcod.ipynb` contains the exploratory training/evaluation workflow.

### Vector bottleneck

`dethcod/vector/` explores a fixed-size continuous bottleneck. A T5 encoder is pooled into a configurable matrix of latent vectors, then projected back into the decoder's model dimension before reconstruction.

The accompanying `VectorDethcod.ipynb` and `dethcod/vector/train.py` contain the corresponding experiments.

## Repository layout

```text
.
├── dethcod/
│   ├── dataset.py
│   ├── token/
│   │   ├── compression.py
│   │   └── decompression.py
│   └── vector/
│       ├── model.py
│       └── train.py
├── TokenDethcod.ipynb
├── VectorDethcod.ipynb
└── Playground.ipynb
```

## Setup

```bash
git clone https://github.com/khoda81/dethcod.git
cd dethcod
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments use Hugging Face Transformers/Datasets, PyTorch, and optional Weights & Biases logging.

## Scope

This repository is research code. The interesting artifact is the compression/modeling formulation and the experiments around bottleneck choice; it should not be treated as a general-purpose compressor or relied on for production data storage.

## License

MIT — see [`LICENSE`](./LICENSE).
