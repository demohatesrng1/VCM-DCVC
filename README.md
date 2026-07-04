# SEC-VCM: Symmetric Entropy-Constrained Video Coding for Machines

> 🎯 **What is SEC-VCM?** A neural video codec that directly aligns video coding with machine vision understanding under visual backbone guidance — achieving state-of-the-art rate-task performance across diverse video understanding tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2510.15347-b31b1b.svg)](https://arxiv.org/abs/2510.15347)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TIP 2026](https://img.shields.io/badge/IEEE-TIP%202026-blue.svg)](https://doi.org/10.1109/TIP.2026.3705185)

**Accepted by IEEE Transactions on Image Processing (TIP), 2026.**

---

## 📖 Overview

As video transmission increasingly serves machine vision systems (MVS) instead of human vision systems (HVS), **video coding for machines (VCM)** has become a critical research topic. Existing VCM methods often bind codecs to specific downstream models, requiring retraining or supervised data, thus limiting generalization in multi-task scenarios.

**SEC-VCM** establishes a **symmetric alignment** between the video codec and visual backbones (VB), allowing the codec to leverage VB's representation capabilities to preserve semantics and discard MVS-irrelevant information.

### Key Contributions

1. **Bi-directional Entropy Constraint (BiEC)** — Ensures symmetry between video decoding and VB encoding by suppressing conditional entropy. This explicitly helps the codec handle semantic information beneficial to MVS while squeezing useless information.

2. **Semantic-Pixel Dual-Path Fusion (SPDF)** — Injects pixel-level priors into the final reconstruction, suppressing artifacts harmful to MVS and improving machine-oriented reconstruction quality.

3. **State-of-the-art results** across multiple video understanding benchmarks:
   - Video Instance Segmentation: **37.4%** bitrate savings vs. VTM
   - Video Object Segmentation: **29.8%** bitrate savings vs. VTM  
   - Object Detection: **46.2%** bitrate savings vs. VTM
   - Multiple Object Tracking: **44.9%** bitrate savings vs. VTM
   - MLLM-based Video Grounding: **97.6%** bitrate savings vs. VTM

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        SEC-VCM Codec                          │
│                                                               │
│  Input x ──► Contextual Encoder ──► Quant ──► Entropy Code   │
│                  │                        │                   │
│                  ▼                        ▼                   │
│            BiEC Module ◄────────── Visual Backbone            │
│          (conditional entropy)     (Mask2Former/Swin)         │
│                  │                        │                   │
│                  ▼                        ▼                   │
│            Semantic Decoder ──► SPDF Fusion ◄── Pixel Recon   │
│                                        │                      │
│                                        ▼                      │
│                                  Output (MVS-optimized)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
sec-vcm/
├── secvcm/                    # Core codec package
│   ├── models/                # SEC-VCM model (DMC), I-frame codec
│   ├── layers/                # CNN layers and building blocks
│   ├── transforms/            # Color space transforms
│   ├── utils/                 # I/O, stream helpers, metrics
│   ├── entropy_models/        # Entropy coding (C++ extension)
│   └── cpp/                   # C++ source for entropy codec
├── train/                     # Training scripts
│   ├── main_ddp.py            # Distributed training entry point
│   ├── video_train.py         # Training loop and logic
│   └── dataset.py             # Data loaders (Vimeo, YouHQ)
├── scripts/                   # Evaluation and utilities
│   ├── test_video.py          # Main evaluation script
│   ├── test.sh                # Test example
│   ├── train.sh               # Train example
│   └── visualize.sh           # Visualization
├── third_party/               # Dependencies
│   ├── mask2former/           # Mask2Former (semantic teacher)
│   ├── dino/                  # DINOv2 (semantic teacher)
│   └── detectron2/            # Detectron2 (required by Mask2Former)
└── pretrain/                  # Flow network pretrained weights
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.3+
- PyTorch 1.11+

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sec-vcm.git
cd sec-vcm

# Create conda environment
conda create -n secvcm python=3.8
conda activate secvcm

# Install PyTorch
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Build C++ entropy coding extension
cd secvcm
mkdir build && cd build
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j
cd ../..
```

### Download Pretrained Models

Download the SEC-VCM pretrained models and place them in the `checkpoint/` directory:

```bash
mkdir -p checkpoint
# Download model weights from the release page (links coming soon)
# Place *.model and *.pth.tar files in checkpoint/
```

For the semantic teachers, download:
- **Mask2Former** (Swin-Tiny backbone) weights → configure path in `cfg.MODEL.WEIGHTS`
- **DINOv2** (ViT-S/14) weights → place in `pretrain/dinov2_vits14_reg4_pretrain.pth`

### Test on a Video

```bash
# Encode and decode a video with SEC-VCM
python scripts/test_video.py \
    --i_frame_model_path ./checkpoint/acmmm2022_image_psnr.pth.tar \
    --model_path ./checkpoint/ckpt-hem10-re2.model \
    --rate_num 4 \
    --test_config ./dataset_config_example.json \
    --cuda True -w 1 \
    --write_stream 0 \
    --output_path output.json
```

### Training

> 📌 SEC-VCM uses a **two-stage training pipeline**. Training is not "from scratch" — you must first complete [Stage 1](#stage-1-dcvc-hem-pre-training-with-lpips-loss) (DCVC-HEM LPIPS pre-training) to obtain a valid pretrained checkpoint, then proceed to [Stage 2](#stage-2-secvcm-semantic-training) (SEC-VCM semantic training). See the full **[Training](#-training)** section below for detailed procedures, phase schedules, and training commands.

## 🏋️ Training

> **⚠️ SEC-VCM training is *not* from scratch.** You must first complete Stage 1 (DCVC-HEM LPIPS pre-training) to obtain a valid pretrained checkpoint before proceeding to Stage 2 (SEC-VCM semantic training).

SEC-VCM employs a **two-stage training strategy** as described in the paper (Section IV-B):

### Stage 1: DCVC-HEM Pre-training with LPIPS Loss

Before training SEC-VCM, you must first obtain a DCVC-HEM [1] base model that has been fine-tuned with **LPIPS (Learned Perceptual Image Patch Similarity)** perceptual loss. This is a critical prerequisite — Stage 2 training depends on this checkpoint.

**Procedure:**
1. Start from the official **DCVC-HEM PSNR-optimized pretrained weights**
2. Fine-tune the model with an **LPIPS-oriented loss function**, using **AlexNet-based LPIPS** as the perceptual distortion term:
   - Loss: $\mathcal{L} = \lambda \cdot R + D_{\text{MSE}} + \alpha \cdot D_{\text{LPIPS}}$ (rate + distortion + perceptual)
   - The AlexNet backbone provides a well-established perceptual feature space for image similarity
3. Train until the model produces reconstructions with good perceptual quality (i.e., convergence on LPIPS metrics)

This stage ensures the base codec has strong perceptual reconstruction capability before semantic guidance is introduced in Stage 2.

We refer readers to the [DCVC-HEM](https://github.com/microsoft/DCVC/tree/main/DCVC-HEM) [1] codebase for the base architecture and training infrastructure. The LPIPS fine-tuning follows the strategy described in the SEC-VCM paper.

### Stage 2: SEC-VCM Semantic Training

Starting from the **LPIPS-fine-tuned DCVC-HEM checkpoint** produced by Stage 1, the full SEC-VCM framework is trained. Stage 2 introduces three novel components on top of the base codec:

| Component | Acronym | Description |
|-----------|---------|-------------|
| **Bi-directional Entropy Constraint** | BiEC | Minimizes conditional entropy between semantic decoder outputs and visual backbone (VB) features at multiple scales (1/4, 1/8, 1/16 resolution), creating a **symmetric alignment** between codec decoding and VB encoding |
| **Semantic-Pixel Dual-Path Fusion** | SPDF | A gated fusion module that adaptively combines semantic-path and pixel-path reconstructions, injecting pixel-level priors into the final output to suppress MVS-harmful artifacts |
| **Multi-Teacher Perception Loss** | — | Combines feature-level perception losses from three complementary teachers: **Mask2Former** (Swin-Tiny, instance-level semantics), **ResNet-18** (CNN-based feature matching), and **DINOv2** (ViT-S/14, transformer-based global features) |

#### Training Schedule (`--train_schedule semantic_v3`)

The training proceeds through three distinct phases with a progressive unfreezing strategy:

| Phase | Epochs | Learning Rate | Trainable Modules | Loss Components |
|-------|--------|---------------|-------------------|-----------------|
| **Warm-up** | 0–2 (×3) | 1×10⁻⁴ | Semantic modules only (BiEC, SPDF, semantic decoder) | MSE + LPIPS (AlexNet) + BiEC + DINOv2 perceptual |
| **Cascaded-1** | 3–5 (×3) | 1×10⁻⁵ | Full model (all parameters unfrozen) | MSE + LPIPS (AlexNet) + BiEC + DINOv2 + Mask2Former + ResNet-18 |
| **Cascaded-2** | 6–8 (×3) | 1×10⁻⁵ | Full model (all parameters unfrozen, GOP=6) | Same as Cascaded-1 |

**Phase details:**
- **Warm-up (epochs 0–2):** Only the newly added semantic components (BiEC, SPDF, semantic decoder) are trained at a higher learning rate (1×10⁻⁴). The base codec weights (from the Stage 1 checkpoint) remain frozen. This allows the semantic modules to learn meaningful representations without destabilizing the base codec.
- **Cascaded-1 (epochs 3–5):** All parameters are unfrozen and trained jointly at a lower learning rate (1×10⁻⁵). All three teacher losses (DINOv2, Mask2Former, ResNet-18) are active alongside BiEC and pixel-space losses, enabling full end-to-end semantic alignment.
- **Cascaded-2 (epochs 6–8):** Continues full-model training with an increased GOP size (GOP=6) for long-range temporal modeling, using the same loss configuration as Cascaded-1.

> 📌 **Note:** The `--stage_extend` parameter acts as a multiplier on the number of epochs per phase. For example, `--stage_extend 2` doubles each phase (Warm-up: 0–4, Cascaded-1: 6–10, Cascaded-2: 12–16). Default is `1`.

#### Training Command

```bash
# ═══════════════════════════════════════════════════════════════
# Stage 1: DCVC-HEM LPIPS Pre-training
# ═══════════════════════════════════════════════════════════════
# Refer to the DCVC-HEM repository for the base training procedure.
# This produces the required checkpoint: ./checkpoint/dcvc-hem-lpips.model

# ═══════════════════════════════════════════════════════════════
# Stage 2: SEC-VCM Semantic Training (4 GPUs)
# ═══════════════════════════════════════════════════════════════
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --use_env train/main_ddp.py \
    --save_path ./checkpoint/secvcm \
    --pretrain ./checkpoint/dcvc-hem-lpips.model \
    --used_data all_vimeo \
    --stage_extend 1 \
    --train_schedule semantic_v3 \
    --save_epoch 4 -b 4 \
    -l ./checkpoint/secvcm/log.txt
```

**Key arguments reference:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--pretrain` | ✅ Yes | Path to the **Stage 1** LPIPS-fine-tuned DCVC-HEM checkpoint. Stage 2 cannot start without this. |
| `--train_schedule semantic_v3` | — | Activates the full semantic training schedule with BiEC, DINOv2, Mask2Former, and ResNet-18 loss terms. |
| `--stage_extend` | — | Multiplier for training epochs per phase. Increase for longer training (e.g., `2` for double). |
| `--save_path` | — | Directory for saving checkpoints and training logs. |
| `--used_data` | — | Dataset identifier string (e.g., `all_vimeo`). Must match your dataset configuration. |
| `-b` / `--batch_size` | — | Batch size per GPU. Adjust based on GPU memory. |
| `--save_epoch` | — | Save a checkpoint every N epochs. |

[1] Li, J., Li, B., & Lu, Y. "Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression." ACM MM 2022.

---

## 🧠 Semantic Teachers

SEC-VCM uses three pre-trained models as semantic teachers during training to compute feature-level losses:

| Teacher | Architecture | Usage |
|---------|-------------|-------|
| **Mask2Former** | Swin-Tiny + Mask2Former head | Primary semantic teacher (BiEC loss) |
| **ResNet-18** | Standard ResNet-18 (ImageNet) | CNN-based perception loss |
| **DINOv2** | ViT-S/14 | Transformer-based perception loss |

These models are **only needed during training**. At inference time, only the SEC-VCM codec itself is required.

---

## 📊 Dataset Preparation

The codec supports PNG frame sequences. Organize your dataset as:

```
dataset/
└── video_name/
    ├── im00001.png
    ├── im00002.png
    ├── im00003.png
    └── ...
```

Or convert YUV to PNG:
```bash
ffmpeg -pix_fmt yuv420p -s WIDTHxHEIGHT -i video.yuv \
    -f image2 video_name/im%05d.png
```

Create a dataset config JSON (see `dataset_config_example.json` for format).

---

## 📈 Results

### Rate-Task Performance on YouTube-VIS 2019

| Metric | vs. VTM (VVC) Bitrate Savings |
|--------|-------------------------------|
| Video Instance Segmentation (Mask2Former) | **37.4%** |
| Video Object Segmentation | **29.8%** |
| Object Detection | **46.2%** |
| Multiple Object Tracking | **44.9%** |
| MLLM-based Video Grounding | **97.6%** |

---

## 🙏 Acknowledgement

This project builds upon:
- [DCVC-HEM](https://github.com/microsoft/DCVC/tree/main/DCVC-HEM) — Base video codec architecture
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) — Learned compression primitives
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) — Semantic teacher model
- [DINOv2](https://github.com/facebookresearch/dinov2) — Visual foundation model teacher
- [Detectron2](https://github.com/facebookresearch/detectron2) — Detection framework

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{sun2025secvcm,
  title   = {Symmetric Entropy-Constrained Video Coding for Machines},
  author  = {Sun, Yuxiao and Liu, Meiqin and Yao, Chao and Tang, Qi and Jin, Jian and Lin, Weisi and Dufaux, Frederic and Zhao, Yao},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  doi     = {10.1109/TIP.2026.3705185}
}

@inproceedings{li2022hybrid,
  title={Hybrid Spatial-Temporal Entropy Modelling for Neural Video Compression},
  author={Li, Jiahao and Li, Bin and Lu, Yan},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE). Third-party components retain their original licenses.
