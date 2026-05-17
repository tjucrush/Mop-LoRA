# MoP-LoRA: Breaking the Linear Bottleneck of Low-Rank Adaptation via Mixture of Paths 🚀

[![Paper](https://img.shields.io/badge/Paper-Under_Review-red.svg)](/)
[![License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0.1%2B-blue.svg)](https://pytorch.org/)

**Status:** Under Review (一区 PR 在投)

This repository contains the official PyTorch implementation of **MoP-LoRA** (Mixture of Paths LoRA).

## 📖 Abstract

Traditional LoRA methods suffer from a heavy linear representation bottleneck. To overcome this critical limitation, we propose a novel re-parameterization Supervised Fine-Tuning (SFT) mechanism: **MoP-LoRA (Mixture of Paths LoRA)**. 

By introducing a **SiLU-gated non-linear activation** and a **multi-path A-matrix architecture** into the low-rank projection space, MoP-LoRA systematically breaks the linear bottleneck.

We comprehensively reproduced and compared MoP-LoRA against 14 mainstream SFT methods (including vanilla LoRA, DoRA, and LongLoRA).

## 🚀 Key Experimental Results

Models evaluated over a **400k mixed instruction-following dataset**:
- `Llama-4-17B`
- `DeepSeek-R1-Distill-7B`
- `Qwen3-8B`
- Multimodal: `LLaVA-OV-1.5-8B`, `Qwen3-VL-8B`

**Improvements against strong baselines:**
| Benchmark / Metric | Absolute Improvement |
|--------------------|----------------------|
| **MATH-500**       | +1.13%              |
| **LiveCodeBench**  | +1.67%              |
| **Multimodal Avg** | +1.52%              |

*Crucially, MoP-LoRA successfully overcomes mode collapse under low-rank regimes.*

## 📂 Repository Structure

- `model.py` - Core PyTorch implementation of the `MoPLoRALayer` and `MoPLinear` wrapper.
- `utils.py` - Scripts to organically inject the MoP-LoRA layer into HuggingFace `transformers` models.
- `train.py` - Example high-performance Trainer setup for models like **DeepSeek-R1** and **Llama**.

## 💻 Quick Start

### 1. Requirements

```bash
pip install torch transformers datasets accelerate
```

### 2. Injecting into Model

```python
from transformers import AutoModelForCausalLM
from utils import inject_mop_lora, print_trainable_parameters

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-7B")

# Inject MoP-LoRA into Query and Value projections
inject_mop_lora(model, target_modules=["q_proj", "v_proj"], r=8, num_paths=4)

print_trainable_parameters(model)
```

### 3. Launch Training

```bash
python train.py --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-7B" --r 8 --num_paths 4
```

## 📜 Citation

Coming soon upon publication.
