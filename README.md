# DLImageSegmentation

Deep learning experiments in image segmentation, built around a from-scratch PyTorch re-implementation of a **piecewise-trained Deep CRF** for semantic segmentation, evaluated on the **NYUDv2** indoor-scene dataset. The repo also contains two supplementary experiments: an instance-segmentation pipeline (Mask Scoring R-CNN + Segment Anything on COCO) and a SIFT-Flow based experiment.

## Project structure

```
DLImageSegmentation/
├── Paper_Implementation_Using_NYUDv2_Dataste.ipynb   # ⭐ Main notebook — Deep CRF semantic segmentation on NYUDv2
├── image-segmentation.ipynb                          # Supplementary: Mask Scoring R-CNN + SAM on COCO (Kaggle)
├── coco.sh                                           # SLURM batch script to download the COCO 2014 dataset
├── setup.txt                                         # Environment / folder layout notes for the MSR-CNN + SAM pipeline
├── SIFT-flow/
│   ├── main.ipynb                                    # SIFT-Flow based segmentation experiment
│   └── Tutorial on accessing SIFT dataset.txt        # Instructions for downloading the SIFT-Flow dataset
└── README.md
```

## Main file: `Paper_Implementation_Using_NYUDv2_Dataste.ipynb`

This notebook implements a **Deep Conditional Random Field (CRF)** for semantic segmentation, following the piecewise-training approach of *Lin et al., "Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation," CVPR 2016*, adapted to the Hugging Face `jagennath-hari/nyuv2` dataset (40-class NYUDv2 labels).

**Architecture**
- **FeatMap-Net** — multi-scale VGG16 backbone (ImageNet-pretrained) with spatial pyramid pooling and a fusion layer that combines features from multiple input scales.
- **Unary-Net** — a shallow 1×1-conv head that produces per-pixel class scores from the fused feature map.
- **Pairwise-Net** — a 1×1-conv head that scores label compatibility between each pixel and its 4-neighbours (up/down/left/right), used to model spatial context.
- **Mean-Field Inference** — an iterative approximate CRF inference module used at test time.

**Training pipeline (3 stages, run end-to-end from `main()`)**
1. **Stage 1 — Unary only:** trains the FCN-style backbone + Unary-Net (FCN baseline).
2. **Stage 2 — + Pairwise:** adds and jointly trains the Pairwise-Net together with the unary potentials, with mean-field CRF inference at evaluation time.
3. **Stage 3 — Fine-tune:** freezes the backbone and fine-tunes the unary/pairwise heads.

Each stage is evaluated with **pixel accuracy** and **mean IoU**, and the notebook includes visualization utilities (`visualize_segmentation`, `visualize_all_stages`) that plot input image / ground truth / prediction side-by-side and compare all three stages on the same samples.

Indicative results from the notebook's own run (40 classes, ~1,014 train / 145 test images):

| Stage | Pixel Accuracy | Mean IoU | Classes Predicted |
|---|---|---|---|
| Stage 1 (Unary only) | 54.07% | 18.98% | 35 |
| Stage 2 (+ Pairwise) | 57.01% | 23.50% | 38 |
| Stage 3 (Fine-tuned) | 57.53% | 24.11% | 39 |

The notebook was originally executed on a CUDA GPU with `torch==2.6.0+cu124`; results will vary with hardware, random seed, and number of epochs.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gubudoos/DLImageSegmentation.git
   cd DLImageSegmentation
   ```

2. **Create and activate a virtual environment** (Python 3.10–3.11 recommended)
   ```bash
   conda create -n dlimageseg python=3.10 -y
   conda activate dlimageseg
   ```
   or with `venv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install PyTorch** (pick the build matching your CUDA version — see [pytorch.org](https://pytorch.org/get-started/locally/)):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install the remaining dependencies**
   ```bash
   pip install datasets numpy matplotlib pillow tqdm jupyter
   ```
   The COCO/Mask-Scoring-R-CNN supplementary notebook (`image-segmentation.ipynb`) additionally needs:
   ```bash
   pip install opencv-python pycocotools yacs tensorboard
   ```

5. **Launch Jupyter and open the main notebook**
   ```bash
   jupyter notebook Paper_Implementation_Using_NYUDv2_Dataste.ipynb
   ```
   The first run downloads the NYUDv2 data automatically via the Hugging Face `datasets` library (`jagennath-hari/nyuv2`), so an internet connection is required.

### Optional: supplementary pipelines

- **`image-segmentation.ipynb`** (Mask Scoring R-CNN + Segment Anything on COCO) requires cloning two external repos and downloading pretrained weights. Follow the step-by-step layout in **`setup.txt`**, and use **`coco.sh`** (a SLURM script) as a reference for downloading the COCO dataset if you're on a Slurm-managed cluster — otherwise adapt the `wget`/`unzip` commands to your environment.
- **`SIFT-flow/main.ipynb`** requires the SIFT-Flow dataset, which is not bundled in the repo. Download it from the link in `SIFT-flow/Tutorial on accessing SIFT dataset.txt` and place it in the same folder as `main.ipynb` before running.

## Package versions

| Package | Version used / recommended |
|---|---|
| Python | 3.10 – 3.11 (main notebook was run on 3.13.5) |
| torch | 2.6.0 (cu124 build) or later, matching your CUDA toolkit |
| torchvision | version paired with your torch install (e.g. 0.21.x for torch 2.6.0) |
| datasets (Hugging Face) | latest |
| numpy | latest |
| matplotlib | latest |
| pillow | latest |
| tqdm | latest |
| opencv-python | latest *(image-segmentation.ipynb only)* |
| pycocotools | latest *(image-segmentation.ipynb only)* |

A CUDA-capable GPU is strongly recommended — the notebooks were developed and timed assuming GPU acceleration (an NVIDIA Tesla P100 was used for the supplementary COCO notebook).

## Notes

- Class labels in the NYUDv2 Hugging Face dataset are stored as 16-bit encoded values; the notebook builds a mapping from these raw values to 40 contiguous class indices (with `255` reserved as the "ignore" label) before training.
- Class weights are computed via inverse frequency to counteract class imbalance in the 40-class label set.
- Training hyperparameters (batch size, learning rates, epoch counts per stage) are set at the top of `main()` in the notebook and can be adjusted directly.
