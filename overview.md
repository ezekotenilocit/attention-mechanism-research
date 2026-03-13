# Attention Mechanisms Overview

## Background
Attention is a technique that allows neural networks to weigh the importance of different parts of an input sequence when producing each output element. It originated in sequence‑to‑sequence models for machine translation and has become fundamental to modern architectures such as Transformers, Vision Transformers, and many multimodal models.

## Core Concepts
- **Query‑Key‑Value**: An attention head computes similarity between a *query* vector and *key* vectors, uses a softmax to obtain weights, and aggregates *value* vectors accordingly.
- **Scaled Dot‑Product**: Dot product of queries and keys scaled by \(\sqrt{d_k}\) to stabilize gradients.
- **Multi‑Head**: Several attention heads run in parallel, each learning different representation sub‑spaces, and their outputs are concatenated.
- **Self‑Attention**: Queries, keys, and values all come from the same sequence, enabling each token to attend to every other token.
- **Encoder‑Decoder Attention**: In seq2seq, the decoder queries attend to keys/values produced by the encoder.

## Architecture
The **Transformer** (Vaswani et al., 2017) replaces recurrence and convolutions with stacks of:
1. **Multi‑Head Self‑Attention** layers (with residual connections and layer normalization).
2. **Position‑wise Feed‑Forward Networks** applied independently to each position.
3. **Positional Encodings** (fixed sinusoidal or learned) to inject order information.

Key architectural parameters (as used in the original paper):
- Model dimension \(d_{model}=512\)
- 8 attention heads, each with \(d_k = d_v = 64\)
- 6 identical encoder and decoder layers
- Feed‑forward inner dimension \(d_{ff}=2048\)

## Applications
- Machine translation (state‑of‑the‑art BLEU scores on WMT‑14 EN‑DE/EN‑FR)
- Language modeling (BERT, GPT, T5)
- Vision (Vision Transformers, ViT‑GANs)
- Speech, multimodal image‑text models (CLIP, DALL‑E)
- Structured tasks (parsing, reasoning, graph neural networks)

## Key Innovations
- **Full attention without recurrence** → massive parallelism.
- **Scaled dot‑product** to address large magnitude of dot products.
- **Multi‑head** mechanism for richer representation learning.
- **Positional encoding** using sinusoids enabling extrapolation to longer sequences.
- **Layer normalization & residual connections** for stable deep training.
- Demonstrated that purely attention‑based models can outperform RNN‑based baselines with far less training time.

---
*Compiled from Wikipedia, the original *Attention Is All You Need* paper, and PyTorch implementation details.*