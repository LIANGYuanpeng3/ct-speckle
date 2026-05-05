import os
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast

# ---------------------------------------------------------
# Make model_unet_small_res.py importable when this file is in src/ablation
# Expected:
#   src/
#     model_unet_small_res.py
#     ablation/
#       predict_selected_angles.py
# ---------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.append(str(SRC_DIR))

from model_unet_small_res import UNetSmallRes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =========================================================
# 0) CONFIG
# =========================================================
ROOT = r"D:\research\niboshi\roi_ablation"
SAVE_ROOT = r"D:\research\niboshi\train_runs"

# Change this to the experiment you want to test.
# Examples:
#   "lf1_to_s_train3"
#   "l_to_s_train3"
#   "lf1_lf2_to_s_train3"
EXPERIMENT_NAME = "lf1_lf2_to_s_train100_maskp95_softloss"
WEIGHT_NAME = "best.pth"   # usually "best.pth"; use "last.pth" if needed

# Choose angles NOT used in TRAIN_INDICES or VAL_INDICES if you want a real test.
TEST_INDICES = [20, 80, 160, 300, 430]

# Fallback settings if config.json is not found.
INPUT_GROUPS = ["lf1"]
GT_GROUP = "s"
TRAIN_INDICES = [39, 234, 390]
H, W = 1856, 1984
P_LOW, P_HIGH = 0.5, 99.5
USE_FLAT = False

PATCH = 256
FULLVAL_OVERLAP = 128
FULLVAL_BATCH = 8
RAW_OUT_MAX = 65535
RAW_DTYPE = np.uint16

MAG_THR = 0.12
MASK_DILATE = 15

FLAT_GROUPS = {
    "l": "flat_l",
    "lf1": "flat_lf1",
    "lf2": "flat_lf2",
    "s": "flat_s",
}


# =========================================================
# 1) IO / UTILS
# =========================================================
def idx_to_name(idx: int) -> str:
    return f"{idx:05d}.RAW"


def find_raw_path(root: str, group: str, name: str) -> str:
    d = os.path.join(root, group)
    candidates = [
        os.path.join(d, name),
        os.path.join(d, name.lower()),
        os.path.join(d, name.upper()),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"RAW not found: group={group}, name={name}, tried={candidates}")


def load_raw_u16(path: str, h: int, w: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=RAW_DTYPE)
    exp = h * w
    if arr.size != exp:
        raise ValueError(f"RAW size mismatch: {path}, got={arr.size}, expected={exp} (H={h}, W={w})")
    return arr.reshape(h, w).astype(np.float32)


def save_raw_u16(path: str, img_u16: np.ndarray) -> None:
    img_u16.astype(np.uint16).tofile(path)


def save_u8(path: str, img01: np.ndarray) -> None:
    u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(path, u8)


def load_flat_mean(root: str, flat_group: str, h: int, w: int) -> np.ndarray:
    p0 = find_raw_path(root, flat_group, "00000.RAW")
    p1 = find_raw_path(root, flat_group, "00001.RAW")
    f0 = load_raw_u16(p0, h, w)
    f1 = load_raw_u16(p1, h, w)
    return np.maximum(0.5 * (f0 + f1), 1.0).astype(np.float32)


def apply_flat(img: np.ndarray, flat: np.ndarray) -> np.ndarray:
    return (img / (flat + 1e-6)).astype(np.float32)


def compute_shared_percentile_stats(imgs: List[np.ndarray], p_low=0.5, p_high=99.5) -> Tuple[float, float]:
    sampled = [im.reshape(-1)[::4] for im in imgs]
    v = np.concatenate(sampled, axis=0)
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    hi = max(hi, lo + 1e-8)
    return lo, hi


def apply_norm(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def sobel_mag(img01: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def make_mask_from_gt(gt01: np.ndarray) -> np.ndarray:
    # 1. 先轻微平滑，避免背景噪声被 Sobel 当成边缘
    gt_blur = cv2.GaussianBlur(gt01.astype(np.float32), (5, 5), 0)

    # 2. Sobel edge magnitude
    mag = sobel_mag(gt_blur)

    # 3. 用 percentile 阈值，而不是固定 0.12
    #    只取最强的 5% 边缘
    thr = np.percentile(mag, 95.0)
    mask = (mag > thr).astype(np.uint8)

    # 4. 适度膨胀，不要用 15，太大了
    k = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)

    # 5. 去掉小噪声区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    min_area = 100
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    return cleaned.astype(np.float32)


def masked_psnr_np(pred01: np.ndarray, gt01: np.ndarray, mask01: np.ndarray) -> float:
    m = mask01.astype(bool)
    if m.sum() < 10:
        m = np.ones_like(mask01, dtype=bool)
    diff = (pred01[m] - gt01[m]).astype(np.float32)
    mse = float(np.mean(diff * diff))
    return float(-10.0 * math.log10(mse + 1e-12))


def masked_mae_np(pred01: np.ndarray, gt01: np.ndarray, mask01: np.ndarray) -> float:
    m = mask01.astype(bool)
    if m.sum() < 10:
        m = np.ones_like(mask01, dtype=bool)
    return float(np.mean(np.abs(pred01[m] - gt01[m])))


def make_hann2d(patch: int) -> np.ndarray:
    w1 = np.hanning(patch).astype(np.float32)
    win = np.outer(w1, w1).astype(np.float32)
    return np.maximum(win, 1e-6)


def save_comparison_png(path: str, input01: np.ndarray, pred01: np.ndarray, gt01: np.ndarray, mask01: np.ndarray) -> None:
    inp = (np.clip(input01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    pred = (np.clip(pred01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    gt = (np.clip(gt01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    diff = np.abs(pred01 - gt01)
    diff = (np.clip(diff * 4.0, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    mask = (np.clip(mask01, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    # Add small labels
    panels = [inp, pred, gt, diff, mask]
    labels = ["input0", "pred", "gt", "absdiff x4", "mask"]
    labeled = []
    for im, lab in zip(panels, labels):
        im3 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im3, lab, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        labeled.append(im3)
    out = cv2.hconcat(labeled)
    cv2.imwrite(path, out)


def load_one_scene(idx: int, input_groups: List[str], gt_group: str,
                   flat_cache: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, str]:
    name = idx_to_name(idx)
    inputs = []
    for group in input_groups:
        p = find_raw_path(ROOT, group, name)
        im = load_raw_u16(p, H, W)
        if USE_FLAT:
            im = apply_flat(im, flat_cache[group])
        inputs.append(im)

    gt_path = find_raw_path(ROOT, gt_group, name)
    gt = load_raw_u16(gt_path, H, W)
    if USE_FLAT:
        gt = apply_flat(gt, flat_cache[gt_group])
    return inputs, gt, name


# =========================================================
# 2) FULL IMAGE INFERENCE
# =========================================================
@torch.no_grad()
def infer_sliding(model, xC01: np.ndarray, device):
    C, HH, WW = xC01.shape
    assert HH == H and WW == W

    step = PATCH - FULLVAL_OVERLAP
    win = make_hann2d(PATCH)

    out = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, H - PATCH + 1, step))
    xs = list(range(0, W - PATCH + 1, step))
    if ys[-1] != H - PATCH:
        ys.append(H - PATCH)
    if xs[-1] != W - PATCH:
        xs.append(W - PATCH)

    coords = [(y, x) for y in ys for x in xs]
    amp_enabled = device.type == "cuda"

    model.eval()
    for i in range(0, len(coords), FULLVAL_BATCH):
        bc = coords[i:i + FULLVAL_BATCH]
        tiles = [xC01[:, y:y + PATCH, x:x + PATCH] for (y, x) in bc]
        tin = torch.from_numpy(np.stack(tiles, axis=0)).to(device)

        with autocast(enabled=amp_enabled):
            tp = model(tin).float().clamp(0, 1)

        tp = tp[:, 0].detach().cpu().numpy().astype(np.float32)

        for (y, x), p in zip(bc, tp):
            out[y:y + PATCH, x:x + PATCH] += p * win
            wsum[y:y + PATCH, x:x + PATCH] += win

    out = out / (wsum + 1e-8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# =========================================================
# 3) MAIN
# =========================================================
def main():
    global INPUT_GROUPS, GT_GROUP, TRAIN_INDICES, H, W, P_LOW, P_HIGH, USE_FLAT
    global PATCH, FULLVAL_OVERLAP, RAW_OUT_MAX, MAG_THR, MASK_DILATE

    exp_dir = os.path.join(SAVE_ROOT, EXPERIMENT_NAME)
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        INPUT_GROUPS = cfg.get("INPUT_GROUPS", INPUT_GROUPS)
        GT_GROUP = cfg.get("GT_GROUP", GT_GROUP)
        TRAIN_INDICES = cfg.get("TRAIN_INDICES", TRAIN_INDICES)
        H = int(cfg.get("H", H))
        W = int(cfg.get("W", W))
        P_LOW = float(cfg.get("P_LOW", P_LOW))
        P_HIGH = float(cfg.get("P_HIGH", P_HIGH))
        USE_FLAT = bool(cfg.get("USE_FLAT", USE_FLAT))
        PATCH = int(cfg.get("PATCH", PATCH))
        MAG_THR = float(cfg.get("MAG_THR", MAG_THR))
        MASK_DILATE = int(cfg.get("MASK_DILATE", MASK_DILATE))
        print(f"[Config] loaded: {config_path}")
    else:
        print(f"[WARN] config.json not found, using fallback config: {config_path}")

    out_dir = os.path.join(exp_dir, f"predict_{WEIGHT_NAME.replace('.pth', '')}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}")
    print(f"[Experiment] {EXPERIMENT_NAME}")
    print(f"[Input] {INPUT_GROUPS} -> {GT_GROUP}, in_ch={len(INPUT_GROUPS)}")
    print(f"[Test] {TEST_INDICES}")

    # Flat cache, only if enabled.
    flat_cache: Dict[str, np.ndarray] = {}
    if USE_FLAT:
        for group in set(INPUT_GROUPS + [GT_GROUP]):
            flat_cache[group] = load_flat_mean(ROOT, FLAT_GROUPS[group], H, W)
        print("[Init] flat-field correction applied.")

    # Recompute normalization from the same train scenes used during training.
    # This is important because train_ablation_selected_angles.py computed lo/hi from TRAIN_INDICES only.
    norm_pool = []
    for idx in TRAIN_INDICES:
        inputs, gt, _ = load_one_scene(idx, INPUT_GROUPS, GT_GROUP, flat_cache)
        norm_pool.extend(inputs)
        norm_pool.append(gt)
    lo, hi = compute_shared_percentile_stats(norm_pool, P_LOW, P_HIGH)
    print(f"[Norm] recomputed from train scenes: lo={lo:.6f}, hi={hi:.6f}")

    # Load model
    weight_path = os.path.join(exp_dir, WEIGHT_NAME)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight not found: {weight_path}")

    model = UNetSmallRes(
        in_ch=len(INPUT_GROUPS),
        out_ch=1,
        base=32,
        use_se=True,
        use_residual=True,
    ).to(device)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[Model] loaded: {weight_path}")

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["index", "name", "masked_psnr", "masked_mae"])

    scores = []
    for idx in TEST_INDICES:
        inputs, gt, name = load_one_scene(idx, INPUT_GROUPS, GT_GROUP, flat_cache)
        inputs01 = [apply_norm(im, lo, hi) for im in inputs]
        gt01 = apply_norm(gt, lo, hi)
        mask01 = make_mask_from_gt(gt01)
        print("mask mean =", mask01.mean(), "mask min/max =", mask01.min(), mask01.max())
        x_full = np.stack(inputs01, axis=0).astype(np.float32)

        print(f"[Predict] {name} ... ", end="", flush=True)
        pred01 = infer_sliding(model, x_full, device=device)

        psnr = masked_psnr_np(pred01, gt01, mask01)
        mae = masked_mae_np(pred01, gt01, mask01)
        scores.append(psnr)
        print(f"masked_PSNR={psnr:.3f} dB, masked_MAE={mae:.6f}")

        stem = os.path.splitext(name)[0]
        # Save normalized visual outputs
        save_u8(os.path.join(out_dir, f"{stem}_input0.png"), inputs01[0])
        for c, im01 in enumerate(inputs01):
            save_u8(os.path.join(out_dir, f"{stem}_input{c}.png"), im01)
        save_u8(os.path.join(out_dir, f"{stem}_pred.png"), pred01)
        save_u8(os.path.join(out_dir, f"{stem}_gt.png"), gt01)
        save_u8(os.path.join(out_dir, f"{stem}_mask.png"), mask01)
        save_u8(os.path.join(out_dir, f"{stem}_absdiff_x4.png"), np.clip(np.abs(pred01 - gt01) * 4.0, 0, 1))
        save_comparison_png(os.path.join(out_dir, f"{stem}_compare.png"), inputs01[0], pred01, gt01, mask01)

        # Save RAW prediction in normalized 0-65535 space for ImageJ check
        pred_u16 = (np.clip(pred01, 0.0, 1.0) * RAW_OUT_MAX + 0.5).astype(np.uint16)
        save_raw_u16(os.path.join(out_dir, f"{stem}_pred_norm.RAW"), pred_u16)

        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([idx, name, f"{psnr:.6f}", f"{mae:.8f}"])

    if scores:
        print(f"\n[Summary] mean masked_PSNR = {float(np.mean(scores)):.3f} dB")
        print(f"[Saved] {out_dir}")
        print(f"[CSV] {summary_path}")


if __name__ == "__main__":
    main()
