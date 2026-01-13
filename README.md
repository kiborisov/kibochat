# kibochat

> A personalized ChatGPT clone built on [nanochat](https://github.com/karpathy/nanochat)

This is **kibochat** (codename "Kibo") - my personal implementation of a full-stack LLM trainer and inference engine, based on Andrej Karpathy's excellent [nanochat](https://github.com/karpathy/nanochat) project. The goal is to train and deploy a personalized language model from scratch for ~$100-$1000.

## Background

I built this project in 48 hours during the [TSFM (Toronto School of Foundation Modelling)](https://www.tsfm.ca/) hackathon - two days of no sleep, fueled by excitement and curiosity about how LLMs actually work under the hood.

The experience was intense: watching the model train for hours, debugging crashes at 3am, implementing checkpoints after losing progress, optimizing batch sizes to fit in VRAM, and finally seeing my custom fine-tuned model respond with its own personality. There's something surreal about training a language model from scratch and then having a conversation with it.

**What I learned:**
- The full LLM training pipeline: pretraining, midtraining, SFT, and RL
- How tokenizers work (BPE) and why they matter for model performance
- Distributed training with PyTorch across multiple GPUs
- The importance of checkpointing (learned this the hard way)
- How synthetic data generation can inject personality into a model
- Infrastructure costs and tradeoffs in ML training

## Results

Trained a d20 (20-layer) Transformer model with custom identity fine-tuning:

- **Training cost**: ~$230 on 8xH100
- **Training time**: ~10 hours total (base + midtraining + SFT)
- **Model size**: 1.9B parameters
- **Custom data**: 1000 synthetic conversations for identity training

The model knows its name (Kibo), who created it, and responds with a distinct personality shaped by the synthetic training data.

## What is this?

kibochat is a complete pipeline for:
- **Tokenization**: Custom BPE tokenizer (Rust + Python)
- **Pretraining**: Base model training on FineWeb dataset
- **Midtraining**: Domain adaptation and knowledge injection
- **SFT**: Supervised fine-tuning for chat capabilities
- **RL**: Reinforcement learning for improved responses
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
# Clone the repository
git clone https://github.com/kiborisov/kibochat.git
cd kibochat

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

### Training (~$100 tier, 4 hours on 8xH100)

```bash
bash speedrun.sh
```

Or run in a screen session:
```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

### Chat with your model

After training completes:

```bash
# Web UI
python -m scripts.chat_web

# CLI
python -m scripts.chat_cli
```

## Environment Variables

For optional features, create a `.env` file (see `.env.example`):

```bash
# For experiment tracking (optional)
WANDB_API_KEY=your_wandb_key

# For HuggingFace datasets (optional)
HUGGINGFACE_TOKEN=your_hf_token

# For synthetic data generation (optional)
OPENROUTER_API_KEY=your_openrouter_key
```

## Project Structure

```
kibochat/
├── nanochat/                    # Core library
│   ├── gpt.py                  # GPT Transformer model
│   ├── tokenizer.py            # BPE tokenizer wrapper
│   ├── engine.py               # Inference engine with KV cache
│   └── ...
├── scripts/                     # Training & inference scripts
│   ├── base_train.py           # Base model training
│   ├── mid_train.py            # Midtraining
│   ├── chat_sft.py             # Supervised fine-tuning
│   ├── chat_rl.py              # Reinforcement learning
│   ├── chat_web.py             # Web UI server
│   └── chat_cli.py             # CLI interface
├── tasks/                       # Evaluation benchmarks
├── rustbpe/                     # Rust BPE tokenizer
├── tests/                       # Unit tests
├── speedrun.sh                  # $100 training script
└── run1000.sh                   # $800 training script
```

## Hardware Requirements

| Tier | Cost | Time | Hardware | Result |
|------|------|------|----------|--------|
| Speedrun | ~$100 | 4h | 8xH100 | d20, kindergartener-level |
| Standard | ~$300 | 12h | 8xH100 | d26, GPT-2 level |
| Full | ~$800 | 33h | 8xH100 | d32, beyond GPT-2 |

The code also runs on:
- Single GPU (8x slower, use gradient accumulation)
- A100 nodes (slightly slower than H100)
- CPU/MPS (for testing only, see `dev/runcpu.sh`)

## Customization

To personalize your model's identity, edit `dev/gen_synthetic_data.py` and generate custom training data:

```bash
export OPENROUTER_API_KEY=your_key
python dev/gen_synthetic_data.py
```

## Tests

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## Acknowledgements

This project was built during the [TSFM (Toronto School of Foundation Modelling)](https://www.tsfm.ca/) hackathon.

Built on top of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy - all credit for the core architecture and training methodology goes to the original project.

Additional thanks to:
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) for training optimizations
- [HuggingFace](https://huggingface.co/) for FineWeb and SmolTalk datasets
- [Modal](https://modal.com/) for GPU compute

## License

MIT (same as nanochat)
