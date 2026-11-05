# LLM_DALI Technical Stack & Mathematical Formulations

## 1. Deep Learning & Systems Stack

| Layer | Component | Version / Specification | Rationale & Operational Role |
| :--- | :--- | :--- | :--- |
| **Compute Hardware** | NVIDIA Tesla T4 | $2\times 15.36\text{ GB}$ (Turing TU104) | Commodity dual-GPU envelope targeted for client reproducibility. |
| **CUDA Driver** | NVIDIA CUDA Toolkit | 12.2 / 12.4 compatible | Native support for tensor cores, half-precision arithmetic, and unified memory. |
| **Deep Learning Framework** | PyTorch | `2.5.1+cu124` | Core autograd engine, tensor manipulation, and multi-stream CUDA execution. |
| **Distributed & Offload** | Hugging Face Accelerate | `1.2.1` | Automated pipeline-parallel layer dispatch (`device_map="auto"`). |
| **Model Architecture** | Transformers | `4.47.1` | Tokenization, causal LM modeling, KV caching, and training loops. |
| **Parameter-Efficient Tuning**| PEFT | `0.14.0` | Multi-adapter mounting, dynamic adapter switching, rank-stabilized LoRA. |
| **Memory Optimization** | PyTorch Memory Allocator | `expandable_segments:True` | Eliminates virtual memory fragmentation on 16GB cards. |
| **Quantization Engine** | BitsAndBytes | `0.45.2` / FP16 Native Fallback | 4-bit NormalFloat (NF4) quantization when compiled; seamless FP16 fallback. |
| **Data & Corpus Engine** | Wikipedia API & Python RegEx | Standard Library / `urllib` | Zero-dependency text extraction and CC BY-SA 4.0 public-domain sourcing. |
| **Evaluation & Statistics** | NumPy, SciPy, Statsmodels | `numpy>=1.26`, `scipy>=1.13` | Exact McNemar's test, 5,000-sample bootstrap confidence intervals. |

---

## 2. Foundation Model Architecture: `Qwen2.5-7B-Instruct`

* **Organization**: Qwen Team / Alibaba Cloud
* **Base Architecture**: Decoder-only Dense Transformer with RoPE (Rotary Position Embeddings), RMSNorm, and SwiGLU activations.
* **Pinned Revision**: `a09a3545` (Strict client contract requirement to guarantee identical weight checksums).
* **Layer Count ($L$)**: 28 transformer decoder blocks.
* **Hidden Dimension ($d_{\text{model}}$)**: 3,584.
* **Intermediate Dimension ($d_{\text{ffn}}$)**: 18,944.
* **Attention Heads ($n_{\text{heads}}$)**: 28 Query Heads, 4 Key/Value Heads (Grouped Query Attention - GQA, ratio 7:1).
* **Vocabulary Size ($|V|$)**: 152,064 tokens.
* **Context Window**: 32,768 tokens (configured at 512–2,048 for fine-tuning).
* **Chat Template & Invariant Delimiters**:
  ```
  <|im_start|>system
  {system_prompt}<|im_end|>
  <|im_start|>user
  {question}<|im_end|>
  <|im_start|>assistant
  {answer}<|im_end|>
  ```

---

## 3. Mathematical Formulations of Continual Learning Algorithms

### 3.1 Low-Rank Adaptation (LoRA)
For a pre-trained frozen projection matrix $W_0 \in \mathbb{R}^{d \times k}$, parameter updates are constrained via low-rank decomposition:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Where:
* $A \in \mathbb{R}^{r \times k} \sim \mathcal{N}\left(0, \frac{1}{r}\right)$ is the down-projection adapter.
* $B \in \mathbb{R}^{d \times r} = 0$ is the up-projection adapter (initialized to zero to preserve identity at step 0).
* $r = 16$ (rank), $\alpha = 32$ (scaling factor $\frac{\alpha}{r} = 2.0$), and dropout $p = 0.05$.
* Target modules: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* Total trainable parameters per adapter: $\approx 40.2\text{M}$ ($0.53\%$ of 7.61B parameters).

---

### 3.2 Token-Level Anchored Kullback-Leibler (KL) Distillation
To prevent weight drift on historical tasks while learning new representations, the student model ($M_2$) is regularized against the frozen teacher ($M_1$ reference adapter) over predicted token distributions:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(y, \hat{y}_{\text{student}}) + \lambda_{\text{KL}} \cdot \mathcal{L}_{\text{distill}}$$

The distillation loss is computed over the logits scaled by temperature $T$:
$$\mathcal{L}_{\text{distill}} = T^2 \cdot D_{\text{KL}}\left(\sigma\left(\frac{z_{\text{ref}}}{T}\right) \;\Big\|\; \sigma\left(\frac{z_{\text{student}}}{T}\right)\right)$$
Expanded over vocabulary $|V|$:
$$\mathcal{L}_{\text{distill}} = T^2 \sum_{i=1}^{|V|} p_i^{\text{ref}} \left(\log p_i^{\text{ref}} - \log q_i^{\text{student}}\right)$$

#### Top-$K$ Subspace Anchoring Optimization
To circumvent computing the full 152,064-dimensional softmax on dual T4 GPUs (saving $\approx 38\%$ activation VRAM), the teacher's top-$K$ ($K=64$) most confident token indices $\Omega_K = \operatorname{arg\,top-}K(z_{\text{ref}})$ are gathered:
$$\mathcal{L}_{\text{anchored\_KL}} = T^2 \cdot D_{\text{KL}}\left(\operatorname{softmax}\left(\frac{z_{\text{ref}}[\Omega_K]}{T}\right) \;\Big\|\; \operatorname{log\_softmax}\left(\frac{z_{\text{student}}[\Omega_K]}{T}\right)\right)$$

---

### 3.3 Experience Replay Buffer Dynamics
For the Protected variant ($M_{2\text{-prot}}$), training batches are synthesized by sampling an exemplar replay buffer $\mathcal{D}_{\text{replay}} \subset \mathcal{D}_A$ alongside new task data $\mathcal{D}_B$:
$$\mathcal{D}_{\text{train}} = \mathcal{D}_B \cup \left(\gamma \cdot \mathcal{D}_{\text{replay}}\right), \quad \gamma = 0.25$$
The combined objective enforces joint retention and acquisition:
$$\mathcal{L}_{\text{prot}} = \mathbb{E}_{(x, y) \sim \mathcal{D}_B} \left[\mathcal{L}_{\text{CE}}(x, y)\right] + \gamma \cdot \mathbb{E}_{(x_A, y_A) \sim \mathcal{D}_{\text{replay}}} \left[\mathcal{L}_{\text{CE}}(x_A, y_A)\right] + \lambda_{\text{KL}} \mathcal{L}_{\text{anchored\_KL}}$$

---

### 3.4 Orthogonal Subspace Regularization (SVD)
Prior task representations span a principal subspace defined by the singular vectors of the adapter weight matrices:
$$W_A = U \Sigma V^T$$
Given cumulative singular value energy threshold $\eta = 0.85$:
$$k = \min \left\{ m \;\Big|\; \frac{\sum_{i=1}^m \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \ge \eta \right\}$$
The basis for the prior subspace is defined by the top-$k$ left singular vectors $P_{\text{prior}} = U_{:, :k} U_{:, :k}^T$. New task adapter weights $W_{\text{new}}$ are penalized for projecting into this subspace:
$$\mathcal{L}_{\text{ortho}} = \frac{1}{|\mathcal{M}|} \sum_{m \in \mathcal{M}} \left\| U_{:, :k}^T W_{\text{new}}^{(m)} \right\|_F^2$$
This ensures weight updates occur strictly in the orthogonal complement $\mathcal{S}_{\text{prior}}^\perp$, mathematically barring retroactive interference.

---

## 4. Hardware Memory Budget Breakdown

```
Dual Tesla T4 VRAM Allocation (30.72 GB Total)
├── GPU 0 (15.36 GB Max)
│   ├── Base Model Parameters (Layers 00-13):  7.60 GB
│   ├── LoRA Trainable Adapter ("default"):     0.08 GB
│   ├── LoRA Reference Adapter ("reference"):   0.08 GB
│   ├── Forward Activation Buffers:            3.10 GB
│   ├── Optimizer States (AdamW on LoRA):      0.32 GB
│   └── Dynamic Working Headroom:              4.18 GB  (FREE)
└── GPU 1 (15.36 GB Max)
    ├── Base Model Parameters (Layers 14-27):  7.60 GB
    ├── LoRA Trainable Adapter ("default"):     0.08 GB
    ├── LoRA Reference Adapter ("reference"):   0.08 GB
    ├── LM Head & Loss Computation:            1.85 GB
    ├── Backward Gradients & Checkpoints:      2.10 GB
    └── Dynamic Working Headroom:              3.65 GB  (FREE)
```

---

## 5. Critical Runtime Flags & Environment Variables

```bash
# Prevent PyTorch VRAM fragmentation on dual 16GB cards
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Prevent torchao version conflicts with PEFT dispatchers
python3 -c "import peft.import_utils; peft.import_utils.is_torchao_available = lambda: False"

# Disable tokenizers parallelism fork warnings
export TOKENIZERS_PARALLELISM="false"
```
