# MASC 515 Assignment 3: MicroGPT Enhancements

This repository contains an enhanced version of the `microgpt` implementation. Following the assignment requirements, four modern Transformer algorithms have been integrated into the core architecture from scratch.

## 1. Gaussian Error Linear Units (GELU)
**Paper:** [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)

**Underlying Idea:** GELU is an activation function used as an alternative to ReLU. Instead of a hard zero cutoff for negative values, GELU weights inputs by their value multiplied by the cumulative distribution function of the Gaussian distribution. This provides a smoother, probabilistic non-linearity that allows negative values to have a small, non-zero gradient, which generally improves training dynamics and convergence in Deep Learning models.

**Mathematical Formulation:**
The exact GELU function is defined as:
$$x \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$
To make this computationally efficient in our code, we use the standard fast approximation:
$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)\right)\right)$$

## 2. Low-Rank Adaptation (LoRA)
**Paper:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**Underlying Idea:**
Large language models have massive weight matrices that are expensive to fine-tune. LoRA hypothesizes that the updates to these matrices during fine-tuning have a low "intrinsic rank". Instead of updating a full pre-trained matrix $W$, LoRA freezes $W$ and trains two smaller matrices, $A$ and $B$, whose product $BA$ matches the dimensions of $W$. This drastically reduces the number of trainable parameters.

**Mathematical Formulation:**
For a pre-trained weight matrix $W \in \mathbb{R}^{d \times k}$, the adapted output $h$ for an input $x$ is:
$$h = Wx + \frac{\alpha}{r} BAx$$
Where $A \in \mathbb{R}^{r \times k}$ is initialized randomly, $B \in \mathbb{R}^{d \times r}$ is initialized to zero (ensuring the initial adaptation is zero), $r$ is the low rank, and $\alpha$ is a scaling factor. In this implementation, LoRA is applied to the Query ($W_q$) and Value ($W_v$) projection matrices.

## 3. Rotary Position Embedding (RoPE)
**Paper:** [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

**Underlying Idea:**
Instead of adding absolute positional embeddings to the token embeddings at the bottom of the network, RoPE encodes relative positional information directly into the multi-head attention mechanism. It achieves this by rotating the query and key representations in a 2D plane by an angle proportional to their absolute position. The dot product between a query and a key then naturally encodes their relative distance.

**Mathematical Formulation:**
For a position $m$ and a feature pair $(x_0, x_1)$, the rotation is applied as:
$$\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}$$
Where $\theta$ varies across the dimensions of the embedding. This operation is applied to all adjacent pairs of features in the Query and Key vectors before computing the attention scores.

## 4. Mixture of Experts (MoE)
**Blog Reference:** [A brief history of MoEs](https://huggingface.co/blog/moe#a-brief-history-of-moes)

**Underlying Idea:**
MoE models replace the standard, dense Feed-Forward Network (MLP block) with a set of specialized sub-networks called "experts," managed by a "router" (or gating network). For every token, the router predicts which expert(s) are best suited to process it. This significantly scales up model capacity (total parameters) while keeping inference time computationally identical to a smaller model, since only a subset of the network is activated per token.

**Implementation Detail:**
This implementation uses a soft gating mechanism over two experts. The router computes a probability distribution over the experts using a Softmax function, and the output is the weighted sum of both expert outputs:
$$y = \sum_{i=1}^{E} G(x)_i \cdot E_i(x)$$
Where $G(x)$ is the output of the gating network and $E_i(x)$ is the output of the $i$-th expert MLP.