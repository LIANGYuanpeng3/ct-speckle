import os
import csv
import numpy as np
import cv2


# =========================================================
# 0) CONFIG
# =========================================================
ROOT = r"D:\research\niboshi\roi_ablation"
GT_GROUP = "s"

H, W = 1856, 1984
RAW_DTYPE = np.uint16

# Pick several representative views.
INDICES = [20, 80, 160, 300, 430]

# If you want to reproduce predict_selected_angles.py normalization more closely,
# set these to the same lo/hi printed by predict script.
# Example:
# NORM_LO = 467.0
# NORM_HI = 1599.0
NORM_LO = None
NORM_HI = None

OUT_DIR = r"D:\research\niboshi\mask_preview"


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
    raise FileNotFoundError(f"RAW not found: group={group}, name={name}")


def load_raw(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=RAW_DTYPE)
    exp = H * W
    if arr.size != exp:
        raise ValueError(f"RAW size mismatch: {path}, got={arr.size}, expected={exp}")
    return arr.reshape(H, W).astype(np.float32)


def normalize_img(img: np.ndarray) -> np.ndarray:
    if NORM_LO is not None and NORM_HI is not None:
        lo, hi = float(NORM_LO), float(NORM_HI)
    else:
        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    hi = max(float(hi), float(lo) + 1e-8)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def save_u8(path: str, img01: np.ndarray) -> None:
    u8 = (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(path, u8)


def sobel_mag(img01: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img01.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img01.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def old_mask(gt01: np.ndarray) -> np.ndarray:
    mag = sobel_mag(gt01)
    mag = mag / (np.percentile(mag, 99) + 1e-8)
    mag = np.clip(mag, 0.0, 1.0)
    mask = (mag > 0.12).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    return mask.astype(np.float32)


def percentile_edge_mask(
    gt01: np.ndarray,
    percentile: float = 95.0,
    blur_ksize: int = 5,
    dilate_ksize: int = 5,
    min_area: int = 100,
) -> np.ndarray:
    if blur_ksize and blur_ksize > 1:
        gt_work = cv2.GaussianBlur(gt01.astype(np.float32), (blur_ksize, blur_ksize), 0)
    else:
        gt_work = gt01.astype(np.float32)

    mag = sobel_mag(gt_work)
    thr = np.percentile(mag, percentile)
    mask = (mag > thr).astype(np.uint8)

    if dilate_ksize and dilate_ksize > 1:
        k = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)

    if min_area and min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1
        mask = cleaned

    return mask.astype(np.float32)


def make_overlay(gt01: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    base = (np.clip(gt01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # Draw mask boundary in white.
    edges = cv2.Canny((mask01 * 255).astype(np.uint8), 50, 150)
    base_bgr[edges > 0] = (255, 255, 255)
    return base_bgr


def add_label(im: np.ndarray, text: str) -> np.ndarray:
    if im.ndim == 2:
        out = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        out = im.copy()
    cv2.putText(out, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


# =========================================================
# 2) MAIN
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mask_methods = [
        ("old_thr012_dil15", lambda x: old_mask(x)),
        ("p93_blur5_dil5", lambda x: percentile_edge_mask(x, percentile=93.0, blur_ksize=5, dilate_ksize=5, min_area=100)),
        ("p95_blur5_dil5", lambda x: percentile_edge_mask(x, percentile=95.0, blur_ksize=5, dilate_ksize=5, min_area=100)),
        ("p97_blur5_dil5", lambda x: percentile_edge_mask(x, percentile=97.0, blur_ksize=5, dilate_ksize=5, min_area=100)),
    ]

    csv_path = os.path.join(OUT_DIR, "mask_stats.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["index", "name", "method", "mask_mean", "mask_pixels"])

        for idx in INDICES:
            name = idx_to_name(idx)
            path = find_raw_path(ROOT, GT_GROUP, name)
            gt = load_raw(path)
            gt01 = normalize_img(gt)

            save_u8(os.path.join(OUT_DIR, f"{idx:05d}_gt_norm.png"), gt01)

            rows = []
            gt_panel = add_label((gt01 * 255).astype(np.uint8), "gt")
            rows.append(gt_panel)

            for method_name, fn in mask_methods:
                mask = fn(gt01)
                mask_mean = float(mask.mean())
                mask_pixels = int(mask.sum())

                print(f"{name} | {method_name}: mean={mask_mean:.6f}, pixels={mask_pixels}")
                wcsv.writerow([idx, name, method_name, f"{mask_mean:.8f}", mask_pixels])

                mask_u8 = add_label((mask * 255).astype(np.uint8), f"{method_name} mean={mask_mean:.3f}")
                overlay = add_label(make_overlay(gt01, mask), "overlay")
                row = cv2.hconcat([mask_u8, overlay])
                rows.append(row)

                cv2.imwrite(os.path.join(OUT_DIR, f"{idx:05d}_{method_name}_mask.png"), (mask * 255).astype(np.uint8))

            target_w = max(r.shape[1] for r in rows)
            rows = [resize_to_width(r, target_w) for r in rows]
            comp = cv2.vconcat(rows)
            cv2.imwrite(os.path.join(OUT_DIR, f"{idx:05d}_mask_compare.png"), comp)

    print(f"\nSaved mask previews to: {OUT_DIR}")
    print(f"Saved stats CSV: {csv_path}")


if __name__ == "__main__":
    main()
