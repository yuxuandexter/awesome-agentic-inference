# Awesome Agentic Inference

## Core Question

How should LLM inference systems evolve when workloads become longer, more adaptive, and more agentic?

## Personal Motivation

- Read the related papers end to end, from foundations to classic inference systems to agentic inference.
- Track top-conference work closely and build an information flywheel through frequent reading, writing, and synthesis.
- Re-implement key ideas with AI assistance to understand implementation details, system logic, and whether the reported experiments hold up.

## Directions

### 1. LLM Disaggregation

| Date | Title | Paper | Code | Conference | Org / Company | Recom |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| 2024.06 | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | [[arXiv]](https://arxiv.org/abs/2407.00079) / [[FAST]](https://www.usenix.org/conference/fast25/presentation/qin) | [[Mooncake]](https://github.com/kvcache-ai/Mooncake) | FAST '25, Best Paper | Moonshot AI, Tsinghua University | ⭐️⭐️ |
| 2025.04 | MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism | [[arXiv]](https://arxiv.org/abs/2504.02263) / [[SIGCOMM]](https://conferences.sigcomm.org/sigcomm/2025/program/papers-info/) | ⚠️ | SIGCOMM '25 | ByteDance, Peking University | ⭐️⭐️ |

### 2. Speculative Decoding in Test Time

| Date | Title | Paper | Code | Conference | Org / Company | Recom |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| 2026.05 | Test-Time Speculation | [[arXiv]](https://arxiv.org/abs/2605.09329) | ⚠️ | arXiv preprint | University of Texas at Austin | ⭐️⭐️ |

### 3. LLM Inference Foundations

| Topic | Resource | Why Read |
|:---|:---:|:---|
| LLM inference overview | [Awesome LLM Inference](https://github.com/xlite-dev/Awesome-LLM-Inference.git) | Use as a broad index, but only pull papers that support the two main directions. |
| Systems readings | [DSC291 Readings](https://haoailab.com/dsc291-s24/) | Build the foundation for serving, scheduling, memory, and distributed inference. |
| Chinese inference notes | [LLM 高性能计算公众号](https://mp.weixin.qq.com/s/dCKqtINt83x_Vw28bBhHJA) | Learn practical inference-system concepts and terminology. |
| Chinese inference notes | [知乎 kaiyuan](https://zhuanlan.zhihu.com/p/2010638958783131701) | Use as background for implementation details and system intuition. |

