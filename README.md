# NEMESIS

**Superpatch-based 3D Medical Image Self-Supervised Pretraining via Noise-Enhanced Dual-Masking**

> IEEE AICAS 2026

---

## Overview

NEMESIS is a self-supervised pretraining framework for 3D CT volumes. It addresses the core challenge of applying Vision Transformers (ViTs) to volumetric medical images — memory constraints and annotation scarcity — through three complementary ideas:

1. **Superpatch processing**: Randomly crop 128³ sub-volumes from full CT scans, enabling ViT-scale pretraining without memory-prohibitive full-volume attention.
2. **Dual-masking (MATB)**: Apply both plane-wise (axial/xy) and axis-wise (depth/z) masking jointly, exploiting the natural anisotropy of CT acquisition.
3. **NEMESIS Tokens (NTs)**: Learnable tokens that attend over unmasked patch slices via multi-head cross-attention, providing a compact summary of visible context for the decoder.

Noise injection during reconstruction further regularises the encoder representations.

### Key results (BTCV organ classification, frozen linear probe)

| Method | AUROC | F1 |
|---|---|---|
| NEMESIS (frozen) | **0.9633** | **0.8791** |
| SuPreM (fine-tuned) | 0.9493 | 0.8602 |
| VoCo (fine-tuned) | 0.9387 | 0.8441 |
| ResNet3D-50 | 0.9312 | 0.8279 |
| Random ViT | 0.8843 | 0.7915 |

NEMESIS frozen encoder outperforms fully fine-tuned competing methods with 32× fewer GFLOPs.

---

## Installation

```bash
git clone https://github.com/hsjung/NEMESIS.git
cd NEMESIS

# Conda (recommended)
conda env create -f environment.yml
conda activate nemesis

# or pip
pip install -r requirements.txt
```

---

## Pretrained Checkpoints

Download from HuggingFace Hub:

```bash
pip install huggingface_hub
huggingface-cli download whilethis/NEMESIS MAE_768_0.5.pt --local-dir pretrained/
```

See [`pretrained/README.md`](pretrained/README.md) for details and alternative download methods.

---

## Dataset Preparation

### Pretraining

NEMESIS was pretrained on a mixed dataset of publicly available CT scans. Prepare a JSON index file in the following format:

```json
{
  "train": [{"image": "/path/to/scan.nii.gz"}, ...],
  "val":   [{"image": "/path/to/scan.nii.gz"}, ...],
  "test":  [{"image": "/path/to/scan.nii.gz"}, ...]
}
```

Public datasets used during pretraining:
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [HNSCC](https://wiki.cancerimagingarchive.net/display/Public/HNSCC)
- [FLARE23](https://codalab.lisn.upsaclay.fr/competitions/12239)
- [TCIA COVID-19](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19)
- [LUNA16](https://luna16.grand-challenge.org/)
- [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)

### Downstream (BTCV classification)

```
data/
  BTCV/
    imagesTr/    # NIfTI CT volumes (*.nii.gz)
    labelsTr/    # NIfTI label maps  (*.nii.gz)
```

Download BTCV from [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and place under `data/BTCV/`.

---

## Pretraining

```bash
python scripts/pretrain.py \
  --config configs/pretrain.yaml \
  --exp_name NEMESIS_pretrain \
  --data_json data/combined_dataset.json \
  --epochs 50 \
  --batch_size 4 \
  --mask_ratio 0.5 \
  --device_ids 0
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/pretrain.yaml` | YAML config file |
| `--exp_name` | required | Experiment name (creates `results/<name>/`) |
| `--data_json` | required | Path to dataset JSON index |
| `--epochs` | 50 | Total training epochs |
| `--batch_size` | 4 | Batch size per GPU |
| `--mask_ratio` | 0.5 | Masking ratio for both axes (plane + axis) |
| `--embed_dim` | 768 | Encoder embedding dimension |
| `--device_ids` | `0` | Comma-separated GPU IDs |
| `--amp` | off | Enable mixed-precision training |
| `--resume` | — | Path to checkpoint to resume from |

---

## Benchmarks

### BTCV organ classification

```bash
# Run all baselines + NEMESIS
bash scripts/run_benchmarks.sh 0   # GPU 0

# Or run a single config
python benchmark/scripts/train_classification.py \
  --config benchmark/configs/btcv_cls_nemesis.yaml \
  --device_ids 0
```

Results are written to `results/<experiment_name>/`.

Benchmark configs available:

| Config | Model |
|---|---|
| `btcv_cls_nemesis.yaml` | NEMESIS (frozen encoder) |
| `btcv_cls_nemesis_finetune.yaml` | NEMESIS (fine-tuned encoder) |
| `btcv_cls_random_vit.yaml` | Random ViT (untrained) |
| `btcv_cls_resnet3d.yaml` | ResNet3D-50 |
| `btcv_cls_voco.yaml` | VoCo (SwinUNETR) |
| `btcv_cls_suprem.yaml` | SuPreM (SwinUNETR) |

For VoCo and SuPreM, download their pretrained weights separately:
- VoCo: `pretrained/VoCo_B_SSL_head.pt` — [VoCo official repo](https://github.com/Luoxd1996/VoCo)
- SuPreM: `pretrained/supervised_suprem_swinunetr_2100.pth` — [SuPreM official repo](https://github.com/MrGiovanni/SuPreM)

---

## Repository Structure

```
NEMESIS/
├── nemesis/                   # Core model package
│   └── models/
│       └── mae.py             # MAEgic3DMAE, MAEgicEncoder, MAEgicDecoder
├── benchmark/                 # Downstream evaluation
│   ├── configs/               # YAML configs per method
│   ├── datasets/              # BTCV, Synapse, KiTS23, MSD Pancreas
│   ├── models/                # Classifier/segmentation heads
│   ├── scripts/               # train_classification.py, train_segmentation.py
│   └── training/              # Trainers, metrics
├── configs/
│   └── pretrain.yaml          # Default pretraining config
├── scripts/
│   ├── pretrain.py            # Pretraining entry point
│   └── run_benchmarks.sh      # Run all benchmarks
├── pretrained/                # Place .pt weights here (see README inside)
├── data/                      # Place datasets here (gitignored)
├── requirements.txt
└── environment.yml
```

---

## Citation

If you use NEMESIS in your research, please cite:

```bibtex
@inproceedings{jung2026nemesis,
  title     = {{NEMESIS}: Superpatch-based 3{D} Medical Image Self-Supervised Pretraining
               via Noise-Enhanced Dual-Masking},
  author    = {Jung, Hyeonseok and others},
  booktitle = {IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)},
  year      = {2026},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
