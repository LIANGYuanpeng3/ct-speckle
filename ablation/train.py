import os
import csv
import json
import random
import time
import math
import sys
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# 让脚本可以读取上一级 src 里的 model_unet_small_res.py
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
sys.path.append(str(SRC_DIR))

from model_unet_small_res import UNetSmallRes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# 0) CONFIG
# =========================================================
ROOT = r"D:\research\niboshi\roi_ablation"
EXPERIMENT_NAME = "lf1_lf2_to_s_train100_maskp95_softloss"
INPUT_GROUPS = ["lf1", "lf2"]
GT_GROUP = "s"

TRAIN_INDICES = list(range(0, 400, 4))  # 约100张
VAL_INDICES   = list(range(402, 462, 4))  # 约15张

H, W = 1856, 1984
RAW_DTYPE = np.uint16

SAVE_ROOT = r"D:\research\niboshi\train_runs"
SAVE_DIR = os.path.join(SAVE_ROOT, EXPERIMENT_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

USE_FLAT = False
FLAT_GROUPS = {
    "l": "flat_l",
    "lf1": "flat_lf1",
    "lf2": "flat_lf2",
    "s": "flat_s",
}

P_LOW = 0.5
P_HIGH = 99.5

SEED = 42
EPOCHS = 20
BATCH = 8
LR = 1e-4
PATCH = 256
STEPS_PER_EPOCH = 250
NUM_WORKERS = 0

MAG_THR = 0.12
MASK_DILATE = 15
PATCH_MIN_MASK_FRAC = 0.03
PATCH_MAX_TRY = 40
FALLBACK_RANDOM_PROB = 0.25

CHAR_EPS = 1e-3
EDGE_LOSS_W = 0.08

FULLVAL_EVERY = 4
FULLVAL_OVERLAP = 64
FULLVAL_BATCH = 8
RAW_OUT_MAX = 65535


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

def load_flat_mean(root: str, flat_group: str) -> np.ndarray:
    p0 = find_raw_path(root, flat_group, "00000.RAW")
    p1 = find_raw_path(root, flat_group, "00001.RAW")
    f0 = load_raw_u16(p0, H, W)
    f1 = load_raw_u16(p1, H, W)
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
    gt_blur = cv2.GaussianBlur(gt01.astype(np.float32), (5, 5), 0)
    mag = sobel_mag(gt_blur)
    thr = np.percentile(mag, 95.0)
    mask = (mag > thr).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)
    
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

def make_hann2d(patch: int) -> np.ndarray:
    w1 = np.hanning(patch).astype(np.float32)
    win = np.outer(w1, w1).astype(np.float32)
    return np.maximum(win, 1e-6)


# =========================================================
# 2) LOSS
# =========================================================
def masked_charbonnier(pred, target, mask, eps=1e-3):
    diff = pred.float() - target.float()
    loss = torch.sqrt(diff * diff + eps * eps) * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return loss.sum() / denom

class WeightedEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)

    def forward(self, pred, target, weight):
        pred = pred.float().clamp(0, 1)
        target = target.float().clamp(0, 1)
        weight = weight.float().clamp(0, 1)

        px = F.conv2d(pred, self.sx, padding=1)
        py = F.conv2d(pred, self.sy, padding=1)
        tx = F.conv2d(target, self.sx, padding=1)
        ty = F.conv2d(target, self.sy, padding=1)

        diff = (px - tx).abs() + (py - ty).abs()
        return (diff * weight).sum() / weight.sum().clamp_min(1.0)


# =========================================================
# 3) SCENE LOADING / DATASET
# =========================================================
@dataclass
class RawScene:
    name: str
    inputs: List[np.ndarray]
    gt: np.ndarray

@dataclass
class NormScene:
    name: str
    inputs01: List[np.ndarray]
    gt01: np.ndarray
    mask01: np.ndarray
    coords: Optional[np.ndarray]

def load_raw_scene_safe(idx: int, flat_cache: Dict[str, np.ndarray]) -> Optional[RawScene]:
    """带有异常捕获的安全场景加载，防止因为极个别坏帧导致程序崩溃"""
    try:
        name = idx_to_name(idx)
        inputs = []
        for group in INPUT_GROUPS:
            p = find_raw_path(ROOT, group, name)
            im = load_raw_u16(p, H, W)
            if USE_FLAT:
                im = apply_flat(im, flat_cache[group])
            inputs.append(im)

        gt_path = find_raw_path(ROOT, GT_GROUP, name)
        gt = load_raw_u16(gt_path, H, W)
        if USE_FLAT:
            gt = apply_flat(gt, flat_cache[GT_GROUP])
        return RawScene(name=name, inputs=inputs, gt=gt)
    except FileNotFoundError as e:
        print(f"[WARN] 跳过缺失文件: {e}")
        return None

def make_norm_scene(raw: RawScene, lo: float, hi: float) -> NormScene:
    inputs01 = [apply_norm(im, lo, hi) for im in raw.inputs]
    gt01 = apply_norm(raw.gt, lo, hi)
    mask01 = make_mask_from_gt(gt01)

    half = PATCH // 2
    y0, y1 = half, H - half
    x0, x1 = half, W - half
    sub = mask01[y0:y1, x0:x1]
    ys, xs = np.where(sub > 0.5)
    if len(ys) == 0:
        coords = None
    else:
        ys = ys + y0
        xs = xs + x0
        coords = np.stack([ys, xs], axis=1).astype(np.int32)
    return NormScene(name=raw.name, inputs01=inputs01, gt01=gt01, mask01=mask01, coords=coords)

class MultiScenePatchDataset(Dataset):
    def __init__(self, scenes: List[NormScene]):
        if not scenes:
            raise ValueError("No scenes provided")
        self.scenes = scenes

    def __len__(self):
        # 抛弃虚假的 4096 长度，直接匹配实际的训练步数
        return STEPS_PER_EPOCH * BATCH

    def _augment(self, x: np.ndarray, y: np.ndarray, m: np.ndarray):
        # 随机翻转和旋转，极大缓解小样本块的过拟合问题
        if random.random() > 0.5:
            x, y, m = np.flip(x, axis=1), np.flip(y, axis=1), np.flip(m, axis=1)
        if random.random() > 0.5:
            x, y, m = np.flip(x, axis=2), np.flip(y, axis=2), np.flip(m, axis=2)
        k = random.randint(0, 3)
        if k > 0:
            x, y, m = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2)), np.rot90(m, k, axes=(1, 2))
        
        # flip / rot90 可能导致连续性问题，需要使用 copy()
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.from_numpy(m.copy())

    def __getitem__(self, idx):
        scene = random.choice(self.scenes)
        half = PATCH // 2

        for _ in range(PATCH_MAX_TRY):
            if scene.coords is not None and random.random() > FALLBACK_RANDOM_PROB:
                iy = random.randrange(len(scene.coords))
                cy, cx = scene.coords[iy]
            else:
                cy = random.randint(half, H - half)
                cx = random.randint(half, W - half)

            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half

            mp = scene.mask01[y0:y1, x0:x1]
            if mp.shape != (PATCH, PATCH):
                continue
            if mp.mean() >= PATCH_MIN_MASK_FRAC:
                xs = [im[y0:y1, x0:x1] for im in scene.inputs01]
                yp = scene.gt01[y0:y1, x0:x1]
                
                x = np.stack(xs, axis=0).astype(np.float32)
                y = yp[None, ...].astype(np.float32)
                m = mp[None, ...].astype(np.float32)
                return self._augment(x, y, m)

        # fallback random patch
        cy = random.randint(half, H - half)
        cx = random.randint(half, W - half)
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half

        xs = [im[y0:y1, x0:x1] for im in scene.inputs01]
        yp = scene.gt01[y0:y1, x0:x1]
        mp = scene.mask01[y0:y1, x0:x1]
        x = np.stack(xs, axis=0).astype(np.float32)
        y = yp[None, ...].astype(np.float32)
        m = mp[None, ...].astype(np.float32)
        return self._augment(x, y, m)


# =========================================================
# 4) FULL IMAGE INFERENCE
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
# 5) VALIDATION
# =========================================================
def validate_full(model, val_scenes: List[NormScene], device, epoch: int) -> float:
    val_dir = os.path.join(SAVE_DIR, "val_outputs")
    os.makedirs(val_dir, exist_ok=True)

    scores = []
    for scene in val_scenes:
        x_full = np.stack(scene.inputs01, axis=0).astype(np.float32)
        pred01 = infer_sliding(model, x_full, device=device)
        psnr = masked_psnr_np(pred01, scene.gt01, scene.mask01)
        scores.append(psnr)

        stem = os.path.splitext(scene.name)[0]
        save_u8(os.path.join(val_dir, f"epoch_{epoch:03d}_{stem}_pred.png"), pred01)
        pred_u16 = (np.clip(pred01, 0.0, 1.0) * RAW_OUT_MAX + 0.5).astype(np.uint16)
        save_raw_u16(os.path.join(val_dir, f"epoch_{epoch:03d}_{stem}_pred.RAW"), pred_u16)

    return float(np.mean(scores)) if scores else float("nan")


# =========================================================
# 6) MAIN
# =========================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    amp_enabled = device.type == "cuda"
    print(f"[Env] device={device}, amp={amp_enabled}")
    print(f"[Experiment] {EXPERIMENT_NAME}")

    # Save config
    config = {
        "ROOT": ROOT,
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "INPUT_GROUPS": INPUT_GROUPS,
        "GT_GROUP": GT_GROUP,
        "TRAIN_INDICES": TRAIN_INDICES,
        "VAL_INDICES": VAL_INDICES,
        "H": H, "W": W,
        "USE_FLAT": USE_FLAT,
        "P_LOW": P_LOW, "P_HIGH": P_HIGH,
        "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
        "PATCH": PATCH, "STEPS_PER_EPOCH": STEPS_PER_EPOCH,
        "MAG_THR": MAG_THR, "MASK_DILATE": MASK_DILATE,
        "PATCH_MIN_MASK_FRAC": PATCH_MIN_MASK_FRAC,
        "PATCH_MAX_TRY": PATCH_MAX_TRY,
        "FALLBACK_RANDOM_PROB": FALLBACK_RANDOM_PROB,
        "EDGE_LOSS_W": EDGE_LOSS_W,
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    flat_cache: Dict[str, np.ndarray] = {}
    if USE_FLAT:
        for group in set(INPUT_GROUPS + [GT_GROUP]):
            flat_group = FLAT_GROUPS[group]
            flat_cache[group] = load_flat_mean(ROOT, flat_group)
        print("[Init] flat-field correction applied.")

    # ---------------------------------------------------------
    # 安全加载并管理内存：生成归一化数据后，立即释放原始内存
    # ---------------------------------------------------------
    print("[Load] loading train scenes...")
    train_raw = []
    for idx in TRAIN_INDICES:
        sc = load_raw_scene_safe(idx, flat_cache)
        if sc: train_raw.append(sc)

    print("[Load] loading val scenes...")
    val_raw = []
    for idx in VAL_INDICES:
        sc = load_raw_scene_safe(idx, flat_cache)
        if sc: val_raw.append(sc)

    norm_pool = []
    for sc in train_raw:
        norm_pool.extend(sc.inputs)
        norm_pool.append(sc.gt)
    lo, hi = compute_shared_percentile_stats(norm_pool, P_LOW, P_HIGH)
    print(f"[Norm] train-shared lo={lo:.6f}, hi={hi:.6f}")

    train_scenes = [make_norm_scene(sc, lo, hi) for sc in train_raw]
    val_scenes = [make_norm_scene(sc, lo, hi) for sc in val_raw]

    # 致命 OOM 修复点：释放原始场景占用的巨大内存并强制垃圾回收
    del train_raw, val_raw, norm_pool
    gc.collect()

    for sc in train_scenes:
        coord_n = 0 if sc.coords is None else len(sc.coords)
        print(f"[Scene train] {sc.name}: mask_mean={sc.mask01.mean():.5f}, coords={coord_n}")

    preview_dir = os.path.join(SAVE_DIR, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    for sc in train_scenes[:5] + val_scenes[:5]: # 仅预览部分，加速启动
        stem = os.path.splitext(sc.name)[0]
        save_u8(os.path.join(preview_dir, f"{stem}_input0.png"), sc.inputs01[0])
        save_u8(os.path.join(preview_dir, f"{stem}_gt.png"), sc.gt01)
        save_u8(os.path.join(preview_dir, f"{stem}_mask.png"), sc.mask01)

    train_ds = MultiScenePatchDataset(train_scenes)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True, # 确保每个 epoch 数据混合均匀
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = UNetSmallRes(
        in_ch=len(INPUT_GROUPS),
        out_ch=1,
        base=32,
        use_se=True,
        use_residual=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler(enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    edge_loss_fn = WeightedEdgeLoss().to(device)

    def criterion(pred, target, mask):
        loss_weight = 0.2 + 0.8 * mask
        lc = masked_charbonnier(pred, target, loss_weight, eps=CHAR_EPS)
        le = edge_loss_fn(pred, target, loss_weight)
        return lc + EDGE_LOSS_W * le

    log_path = os.path.join(SAVE_DIR, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["epoch", "train_loss", "val_full_psnr_mean", "lr"])

    print(f"[Train] epochs={EPOCHS}, steps_per_epoch={STEPS_PER_EPOCH}, batch={BATCH}, patch={PATCH}")
    best_psnr = -1e9
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        # 直接利用 DataLoader 的迭代器特性，抛弃手动 StopIteration 控制
        for step, (x, y, m) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                pred = model(x)
                loss = criterion(pred, y, m)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())

        scheduler.step()
        train_loss = running / STEPS_PER_EPOCH
        val_psnr = float("nan")

        if epoch == 1 or epoch % 2 == 0 or epoch == EPOCHS:
            model.eval()
            sc = train_scenes[0]
            half = PATCH // 2
            cy, cx = H // 2, W // 2
            vx_np = np.stack([im[cy-half:cy+half, cx-half:cx+half] for im in sc.inputs01], axis=0)[None, ...]
            with torch.no_grad():
                vx = torch.from_numpy(vx_np.astype(np.float32)).to(device)
                with autocast(enabled=amp_enabled):
                    pv = model(vx).float().clamp(0, 1)
            pv_np = pv[0, 0].detach().cpu().numpy()
            save_u8(os.path.join(SAVE_DIR, f"patch_pred_epoch_{epoch:03d}.png"), pv_np)

        if epoch % FULLVAL_EVERY == 0 or epoch == EPOCHS:
            val_psnr = validate_full(model, val_scenes, device, epoch)
            print(f"[Epoch {epoch:03d}] loss={train_loss:.6f} | val_full_psnr_mean={val_psnr:.3f} dB")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
                print(f"           [Best] saved best.pth ({best_psnr:.3f} dB)")
        else:
            print(f"[Epoch {epoch:03d}] loss={train_loss:.6f}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last.pth"))
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([epoch, f"{train_loss:.8f}", f"{val_psnr:.6f}", opt.param_groups[0]["lr"]])

    print(f"[Done] elapsed={(time.time() - start) / 60:.1f} min")
    print(f"[Saved] {SAVE_DIR}")

if __name__ == "__main__":
    main()