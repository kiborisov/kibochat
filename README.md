# kibochat

> A personalized ChatGPT clone built on [nanochat](https://github.com/karpathy/nanochat)

This is **kibochat** (codename "Kibo") - my personal LLM trained from scratch during a 48-hour hackathon. Built on Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) framework.

## Background

I built this at the [TSFM (Toronto School of Foundation Modelling)](https://www.tsfm.ca/) hackathon. Two days, no sleep, one goal: train my own language model from scratch and actually understand what's happening under the hood.

The first few runs crashed. I'd watch the loss curve for an hour, step away for coffee, come back to a CUDA out-of-memory error. No checkpoints saved. Progress gone. That's when I learned why checkpointing matters - not from a textbook, but from losing 3 hours of training at 2am.

By the second night, I had the pipeline stable. Watching the validation loss tick down from 0.827 to 0.816 over thousands of steps - that's when it clicked. This wasn't magic. It was matrix multiplications, gradient updates, and a lot of patience.

The final piece was identity training. I generated 1000 synthetic conversations teaching the model who it is - Kibo, built by me, running on Modal's H100s. When I finally asked it "who are you?" and it answered correctly, that was the moment.

**What I learned:**
- The full LLM training pipeline: pretraining → midtraining → SFT
- Why checkpointing isn't optional (learned the hard way)
- Distributed training with PyTorch across 8 GPUs
- How synthetic data shapes model personality
- The relationship between loss curves and actual model capability

## Training Results

### Final Metrics

| Stage | Steps | Final Loss | Key Results |
|-------|-------|------------|-------------|
| Base Pretraining | 21,400 | 2.68 | CORE: 0.211, BPB: 0.816 |
| Midtraining | 810 | 1.24 | BPB: 0.397 |
| SFT | 700 | 0.48 | ARC-Easy: 47.9%, MMLU: 35.5% |

### Infrastructure

- **Compute**: [Modal](https://modal.com/) cloud GPUs (8xH100)
- **Total cost**: ~$230 (from $500 TSFM credits)
- **Training time**: ~12 hours total
- **Model**: 1.9B parameters, d20 (20-layer Transformer)

---

## Training Curves

All metrics tracked on [Weights & Biases](https://wandb.ai/kiborisov-asc42-com).

### Stage 1: Base Pretraining

The foundation - training on FineWeb dataset to learn language patterns.

**Training Loss** - Started high, steadily decreased as the model learned
![Base Training Loss](assets/base_train_loss.png)

**Validation BPB (Bits Per Byte)** - The real measure of generalization: 0.827 → 0.816
![Base Validation BPB](assets/base_val_bpb.png)

**GPU Utilization (MFU)** - Sustained ~21% model FLOP utilization across 8xH100
![Base MFU](assets/base_train_mfu.png)

**Throughput** - ~240K tokens/sec training speed
![Base Throughput](assets/base_train_throughput.png)

---

### Stage 2: Midtraining

Domain adaptation with chat-style data and custom identity conversations.

**Training Loss** - Sharp drop from 1.9 → 1.24 as model adapts to conversational format
![Mid Training Loss](assets/mid_train_loss.png)

**Validation BPB** - Dramatic improvement: 0.70 → 0.40
![Mid Validation BPB](assets/mid_val_bpb.png)

---

### Stage 3: Supervised Fine-Tuning (SFT)

Final polish - teaching the model to follow instructions and maintain identity.

**Training Loss** - Noisy but trending down (expected for small-batch SFT)
![SFT Training Loss](assets/sft_train_loss.png)

**Validation Loss** - Stable convergence around 1.0
![SFT Validation Loss](assets/sft_val_loss.png)

**MMLU Accuracy** - Benchmark performance improving: 32.8% → 35.5%
![SFT MMLU](assets/sft_mmlu_acc.png)

---

## What is this?

kibochat is a complete pipeline for training and deploying a personalized LLM:

- **Tokenization**: Custom BPE tokenizer (Rust + Python)
- **Pretraining**: Base model training on FineWeb dataset
- **Midtraining**: Domain adaptation and identity injection
- **SFT**: Supervised fine-tuning for chat capabilities
- **Inference**: Efficient KV-cache inference engine
- **Deployment**: Web UI and CLI interfaces

## Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- GPU with 80GB VRAM (8xH100 recommended) for full training
- For inference only: Any CUDA GPU or CPU/MPS

### Setup

```bash
git clone https://github.com/kiborisov/kibochat.git
cd kibochat
uv sync
source .venv/bin/activate
```

### Training

```bash
# Full training (~$100, 4 hours on 8xH100)
bash speedrun.sh

# Or in a screen session
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

### Chat with your model

```bash
# Web UI
python -m scripts.chat_web

# CLI
python -m scripts.chat_cli
```

## Project Structure

```
kibochat/
├── nanochat/           # Core library (model, tokenizer, inference)
├── scripts/            # Training & inference scripts
├── tasks/              # Evaluation benchmarks (ARC, MMLU, GSM8K, HumanEval)
├── rustbpe/            # Rust BPE tokenizer
├── tests/              # Unit tests
├── speedrun.sh         # $100 training script
└── run1000.sh          # $800 training script
```

## Customization

To give your model a custom identity:

```bash
export OPENROUTER_API_KEY=your_key
python dev/gen_synthetic_data.py
```

This generates synthetic conversations that get mixed into midtraining and SFT.

## Acknowledgements

Built during [TSFM (Toronto School of Foundation Modelling)](https://www.tsfm.ca/) hackathon.

Core architecture from [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.

Thanks to:
- [Modal](https://modal.com/) for $500 in GPU credits via TSFM
- [HuggingFace](https://huggingface.co/) for FineWeb and SmolTalk datasets
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) for training optimizations

## License

MIT
