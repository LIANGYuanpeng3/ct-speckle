import os
import csv
import math
import gc
import numpy as np
import cv2


# =========================================================
# 0) CONFIG
# =========================================================
ROOT = r"D:\research\niboshi\roi_ablation"

INPUT_GROUP = "l"
GT_GROUP = "s"

# Use the same test views as your current predict_selected_angles.py run.
TEST_INDICES = [20, 80, 160, 300, 430]

H, W = 1856, 1984
RAW_DTYPE = np.uint16

# Use the same training views as lf1_lf2_to_s_train100 for normalization.
# This keeps the scale closer to your model evaluation setup.
NORM_INDICES = list(range(0, 400, 4))

P_LOW = 0.5
P_HIGH = 99.5

OUT_DIR = r"D:\research\niboshi\train_runs\l_s_direct_baseline"
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


def load_raw(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=RAW_DTYPE)
    exp = H * W
    if arr.size != exp:
        raise ValueError(f"RAW size mismatch: {path}, got={arr.size}, expected={exp}")
    return arr.reshape(H, W).astype(np.float32)


def save_u8(path: str, img01: np.ndarray) -> None:
    u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(path, u8)


def save_raw_u16(path: str, img01: np.ndarray) -> None:
    u16 = (np.clip(img01, 0, 1) * RAW_OUT_MAX + 0.5).astype(np.uint16)
    u16.tofile(path)


def compute_shared_percentile_stats_from_indices(indices, p_low=0.5, p_high=99.5):
    """
    Memory-friendly percentile estimation.
    It samples every 8th pixel from L and S images, instead of storing all full images.
    """
    sampled = []
    for idx in indices:
        name = idx_to_name(idx)
        l = load_raw(find_raw_path(ROOT, INPUT_GROUP, name))
        s = load_raw(find_raw_path(ROOT, GT_GROUP, name))
        sampled.append(l.reshape(-1)[::8])
        sampled.append(s.reshape(-1)[::8])

    v = np.concatenate(sampled, axis=0)
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    hi = max(hi, lo + 1e-8)

    del sampled
    del v
    gc.collect()

    return lo, hi


def apply_norm(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def sobel_mag(img01: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img01.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def make_mask_from_gt(gt01: np.ndarray) -> np.ndarray:
    """
    Corrected evaluation mask:
      Gaussian blur 5x5
      Sobel magnitude
      top 5% edge responses
      dilation 5x5
      connected component cleanup
    """
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


def masked_psnr(a01: np.ndarray, b01: np.ndarray, mask01: np.ndarray) -> float:
    m = mask01.astype(bool)
    if m.sum() < 10:
        m = np.ones_like(mask01, dtype=bool)
    mse = float(np.mean((a01[m] - b01[m]) ** 2))
    return float(-10.0 * math.log10(mse + 1e-12))


def masked_mae(a01: np.ndarray, b01: np.ndarray, mask01: np.ndarray) -> float:
    m = mask01.astype(bool)
    if m.sum() < 10:
        m = np.ones_like(mask01, dtype=bool)
    return float(np.mean(np.abs(a01[m] - b01[m])))


def whole_psnr(a01: np.ndarray, b01: np.ndarray) -> float:
    mse = float(np.mean((a01 - b01) ** 2))
    return float(-10.0 * math.log10(mse + 1e-12))


def whole_mae(a01: np.ndarray, b01: np.ndarray) -> float:
    return float(np.mean(np.abs(a01 - b01)))


def fit_affine(src01: np.ndarray, tgt01: np.ndarray, mask01: np.ndarray = None, eps=1e-8):
    """
    Fit alpha, beta so that alpha * src + beta approximates tgt.
    If mask is given, fit only on masked region.
    """
    if mask01 is not None:
        m = mask01.astype(bool)
        if m.sum() < 10:
            m = np.ones_like(mask01, dtype=bool)
        s = src01[m].reshape(-1).astype(np.float32)
        t = tgt01[m].reshape(-1).astype(np.float32)
    else:
        s = src01.reshape(-1).astype(np.float32)
        t = tgt01.reshape(-1).astype(np.float32)

    s_mean = float(s.mean())
    t_mean = float(t.mean())
    s_var = float(((s - s_mean) ** 2).mean())

    if s_var < eps:
        return 1.0, t_mean - s_mean

    cov = float(((s - s_mean) * (t - t_mean)).mean())
    alpha = cov / (s_var + eps)
    beta = t_mean - alpha * s_mean
    return float(alpha), float(beta)


def apply_affine(src01: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.clip(alpha * src01 + beta, 0.0, 1.0).astype(np.float32)


def add_label(im: np.ndarray, text: str) -> np.ndarray:
    if im.ndim == 2:
        out = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        out = im.copy()
    cv2.putText(out, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_compare_png(path: str, l01: np.ndarray, l_fit01: np.ndarray, s01: np.ndarray, mask01: np.ndarray) -> None:
    direct_diff = np.abs(l01 - s01)
    fit_diff = np.abs(l_fit01 - s01)

    panels = [
        add_label((l01 * 255.0 + 0.5).astype(np.uint8), "L direct"),
        add_label((l_fit01 * 255.0 + 0.5).astype(np.uint8), "L affine-fit"),
        add_label((s01 * 255.0 + 0.5).astype(np.uint8), "S gt"),
        add_label((np.clip(direct_diff * 4.0, 0, 1) * 255.0 + 0.5).astype(np.uint8), "direct diff x4"),
        add_label((np.clip(fit_diff * 4.0, 0, 1) * 255.0 + 0.5).astype(np.uint8), "fit diff x4"),
        add_label((mask01 * 255.0 + 0.5).astype(np.uint8), "mask"),
    ]

    out = cv2.hconcat(panels)
    cv2.imwrite(path, out)


# =========================================================
# 2) MAIN
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[Norm] computing shared lo/hi from NORM_INDICES...")
    lo, hi = compute_shared_percentile_stats_from_indices(NORM_INDICES, P_LOW, P_HIGH)

    print(f"[Norm] lo={lo:.6f}, hi={hi:.6f}")
    print(f"[Compare] {INPUT_GROUP} vs {GT_GROUP}")
    print(f"[Test] {TEST_INDICES}")

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "name",
            "direct_masked_psnr", "direct_masked_mae",
            "direct_whole_psnr", "direct_whole_mae",
            "fit_masked_psnr", "fit_masked_mae",
            "fit_whole_psnr", "fit_whole_mae",
            "fit_alpha", "fit_beta",
            "mask_mean",
        ])

    direct_masked_psnrs = []
    direct_masked_maes = []
    direct_whole_psnrs = []
    direct_whole_maes = []

    fit_masked_psnrs = []
    fit_masked_maes = []
    fit_whole_psnrs = []
    fit_whole_maes = []

    for idx in TEST_INDICES:
        name = idx_to_name(idx)
        l_raw = load_raw(find_raw_path(ROOT, INPUT_GROUP, name))
        s_raw = load_raw(find_raw_path(ROOT, GT_GROUP, name))

        l01 = apply_norm(l_raw, lo, hi)
        s01 = apply_norm(s_raw, lo, hi)
        mask01 = make_mask_from_gt(s01)

        # Direct comparison
        d_m_psnr = masked_psnr(l01, s01, mask01)
        d_m_mae = masked_mae(l01, s01, mask01)
        d_w_psnr = whole_psnr(l01, s01)
        d_w_mae = whole_mae(l01, s01)

        # Affine intensity-fit baseline
        alpha, beta = fit_affine(l01, s01, mask01)
        l_fit01 = apply_affine(l01, alpha, beta)

        f_m_psnr = masked_psnr(l_fit01, s01, mask01)
        f_m_mae = masked_mae(l_fit01, s01, mask01)
        f_w_psnr = whole_psnr(l_fit01, s01)
        f_w_mae = whole_mae(l_fit01, s01)

        direct_masked_psnrs.append(d_m_psnr)
        direct_masked_maes.append(d_m_mae)
        direct_whole_psnrs.append(d_w_psnr)
        direct_whole_maes.append(d_w_mae)

        fit_masked_psnrs.append(f_m_psnr)
        fit_masked_maes.append(f_m_mae)
        fit_whole_psnrs.append(f_w_psnr)
        fit_whole_maes.append(f_w_mae)

        stem = os.path.splitext(name)[0]
        save_u8(os.path.join(OUT_DIR, f"{stem}_l_direct.png"), l01)
        save_u8(os.path.join(OUT_DIR, f"{stem}_l_affine_fit.png"), l_fit01)
        save_u8(os.path.join(OUT_DIR, f"{stem}_s_gt.png"), s01)
        save_u8(os.path.join(OUT_DIR, f"{stem}_mask.png"), mask01)
        save_u8(os.path.join(OUT_DIR, f"{stem}_direct_absdiff_x4.png"), np.clip(np.abs(l01 - s01) * 4.0, 0, 1))
        save_u8(os.path.join(OUT_DIR, f"{stem}_fit_absdiff_x4.png"), np.clip(np.abs(l_fit01 - s01) * 4.0, 0, 1))
        save_compare_png(os.path.join(OUT_DIR, f"{stem}_compare.png"), l01, l_fit01, s01, mask01)

        save_raw_u16(os.path.join(OUT_DIR, f"{stem}_l_direct_norm.RAW"), l01)
        save_raw_u16(os.path.join(OUT_DIR, f"{stem}_l_affine_fit_norm.RAW"), l_fit01)

        with open(summary_path, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                idx, name,
                f"{d_m_psnr:.6f}", f"{d_m_mae:.8f}",
                f"{d_w_psnr:.6f}", f"{d_w_mae:.8f}",
                f"{f_m_psnr:.6f}", f"{f_m_mae:.8f}",
                f"{f_w_psnr:.6f}", f"{f_w_mae:.8f}",
                f"{alpha:.8f}", f"{beta:.8f}",
                f"{float(mask01.mean()):.8f}",
            ])

        print(
            f"[{name}] "
            f"direct masked_PSNR={d_m_psnr:.3f} dB, "
            f"fit masked_PSNR={f_m_psnr:.3f} dB, "
            f"alpha={alpha:.4f}, beta={beta:.4f}, "
            f"mask_mean={mask01.mean():.4f}"
        )

    print("\n[Summary]")
    print(f"direct mean masked_PSNR = {float(np.mean(direct_masked_psnrs)):.3f} dB")
    print(f"direct mean masked_MAE  = {float(np.mean(direct_masked_maes)):.6f}")
    print(f"direct mean whole_PSNR  = {float(np.mean(direct_whole_psnrs)):.3f} dB")
    print(f"direct mean whole_MAE   = {float(np.mean(direct_whole_maes)):.6f}")

    print(f"fit mean masked_PSNR    = {float(np.mean(fit_masked_psnrs)):.3f} dB")
    print(f"fit mean masked_MAE     = {float(np.mean(fit_masked_maes)):.6f}")
    print(f"fit mean whole_PSNR     = {float(np.mean(fit_whole_psnrs)):.3f} dB")
    print(f"fit mean whole_MAE      = {float(np.mean(fit_whole_maes)):.6f}")

    print(f"[Saved] {OUT_DIR}")
    print(f"[CSV] {summary_path}")


if __name__ == "__main__":
    main()
