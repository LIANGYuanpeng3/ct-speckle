# X-ray CT Image Quality Restoration with Speckle Patterns

## 日本語概要

大焦点X線源で撮影されたX線CT透過画像を対象に、スペックルパターンを利用して小焦点X線源に近い画質へ復元する研究プロジェクトです。PythonによるRAW画像の読み込み・正規化、ROI抽出、OpenCVを用いたマスク処理、PyTorchによる残差U-Netモデルの学習・推論、masked PSNR / MAEによる評価、比較表・図の作成までを実装しています。

## Project Overview

This project studies image-quality restoration for X-ray CT transmission images captured with a large-focus X-ray source. The goal is to use speckle-pattern information and deep learning to recover image quality closer to small-focus X-ray source images.

The current repository focuses on RAW CT transmission image handling, ROI-based ablation experiments, image alignment / masking utilities, U-Net-style restoration models, quantitative evaluation, and visualization scripts for comparison tables and figures.

## Main Work

- Load and normalize 16-bit RAW X-ray CT transmission images with Python and NumPy
- Build paired datasets between large-focus / speckle-assisted inputs and small-focus targets
- Extract ROI patches and generate masks for structure-aware training
- Use OpenCV for preprocessing, mask generation, image comparison, and output export
- Train a residual U-Net model implemented in PyTorch
- Evaluate restoration quality with masked PSNR / MAE and ablation summaries
- Generate comparison tables and figures with Matplotlib

## Repository Structure

```text
ct-speckle/
  model_unet_small_res.py
  ablation/
    make_pairs_ablation_l_lf1_lf2_s.py
    train.py
    train_ablation_selected_angles.py
    predict_selected_angles.py
    make_ablation_tables.py
    compare_l_vs_s_direct.py
    compare_l_vs_s_direct_affine.py
    preview_mask_versions.py
    preview_ablation_roi_lf1_lf2.py
    crop_ablation_roi_l_lf1_lf2_s.py
```

## Model

The main model is `UNetSmallRes`, a lightweight residual U-Net with:

- GroupNorm for small-batch stability
- SiLU activations
- residual blocks
- optional SE channel attention
- residual learning from the input image
- sliding-window inference for full-size CT images

## Evaluation

The ablation workflow compares different input settings such as:

- `L -> S`: large-focus image to small-focus target
- `LF1 -> S`: large-focus image with speckle pattern 1
- `LF2 -> S`: large-focus image with speckle pattern 2
- `LF1 + LF2 -> S`: multi-speckle input restoration

Evaluation scripts export per-view and summary results including masked PSNR and masked MAE. Table and bar-plot scripts are included for report-ready visualization.

## Technologies

- Python
- NumPy
- OpenCV
- PyTorch
- Matplotlib
- RAW image processing
- U-Net / residual CNN
- PSNR / MAE based image-quality evaluation

## Portfolio Summary

Large-focus X-ray CT transmission images were restored toward small-focus image quality by using speckle-pattern inputs and a PyTorch residual U-Net. The workflow covers RAW image loading, normalization, ROI and mask generation, ablation dataset construction, model training, full-image inference, quantitative evaluation, and comparison-table visualization.
