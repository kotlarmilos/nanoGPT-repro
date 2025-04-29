# NanoGPT-repro

A minimal repro of ChatGPT based on https://github.com/karpathy/nanoGPT.

This repo demonstrates how to build, train, and sample from a small transformer-based language model with multi-block attention.

---

## Features

- Token & positional embeddings
- Multihead transformer blocks
- Simple feed-forward residual blocks with normalization
- Multinomial sampling

---


## Requirements

- Python 3.8+
- PyTorch 1.12+

Install dependencies:

```bash
pip install torch
```

## Hyperparameters

| Name            | Default | Description                                   |
| --------------- | ------- | --------------------------------------------- |
| `batch_size`    | 32      | Parallel sequences per iteration              |
| `context_size`  | 256     | Maximum context length                        |
| `n_embed`       | 512     | Embedding dimension                           |
| `n_transformers`| 16      | Number of transformer blocks                  |
| `n_iters`       | 15000   | Total training iterations                     |
| `learning_rate` | 1e-3    | Adam optimizer learning rate                  |
| `log_interval`  | 100     | Steps between training logs and sample outputs|

---

## TODOs
- [ ] Replace char-level tokenization with `tiktoken` / `sentencepiece`
- [ ] Add model checkpointing & resume capability
- [ ] Implement learning rate scheduler
- [ ] Support multi-GPU / distributed training
- [ ] Reduce precision

## Results

