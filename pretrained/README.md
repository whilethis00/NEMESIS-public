# Pretrained Checkpoints

NEMESIS pretrained weights are hosted on HuggingFace Hub.

## Available Checkpoints

| Checkpoint | embed_dim | depth | mask_ratio | Description |
|---|---|---|---|---|
| `MAE_768_0.5.pt` | 768 | 6 | 0.5 | Main NEMESIS model (used in paper) |

## Download

### Using Python (recommended)

```python
from huggingface_hub import hf_hub_download

# Download main checkpoint
path = hf_hub_download(
    repo_id="hsjung/NEMESIS",
    filename="MAE_768_0.5.pt",
    local_dir="pretrained/",
)
print(f"Saved to {path}")
```

### Using the HuggingFace CLI

```bash
pip install huggingface_hub
huggingface-cli download hsjung/NEMESIS MAE_768_0.5.pt --local-dir pretrained/
```

### Manual download

Visit https://huggingface.co/hsjung/NEMESIS and download `MAE_768_0.5.pt` into this folder.

## Checkpoint format

Each `.pt` file is a standard PyTorch checkpoint saved with `torch.save`. It contains:

```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,   # MAEgic3DMAE weights
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "best_val_loss": float,
    "config": dict,                    # training hyperparameters
}
```

To load the encoder only:

```python
import torch
from nemesis.models.mae import MAEgic3DMAE

ckpt = torch.load("pretrained/MAE_768_0.5.pt", map_location="cpu")
model = MAEgic3DMAE(
    embed_dim=768, depth=6, num_heads=8,
    decoder_embed_dim=128, decoder_depth=3,
    num_maegic_tokens=8,
)
model.load_state_dict(ckpt["model_state_dict"])
encoder = model.encoder   # MAEgicEncoder
```
