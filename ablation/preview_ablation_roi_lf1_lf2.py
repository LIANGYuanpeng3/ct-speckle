import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# ROI preview script.
#
# Purpose:
#   Generate PNG previews from lf1/lf2 so you can tune X0, Y0, W, H.
#
# Usage:
#   1. Set ROOT and ROI below.
#   2. Run this script.
#   3. Check PNGs in OUT_DIR.
#   4. Adjust X0, Y0, W, H until the ROI is correct.
#   5. Copy the final X0, Y0, W, H into crop_ablation_roi.py.
# ============================================================

ROOT = r"D:\research\niboshi"
OUT_DIR = r"D:\research\niboshi\roi_preview"

GROUPS_FOR_PREVIEW = ["lf1", "lf2"]

IMG_H, IMG_W = 2944, 2352
DTYPE = np.uint16

# TODO: adjust this repeatedly
X0, Y0 = 56, 478
W,  H  = 1984, 1856
X1, Y1 = X0 + W, Y0 + H


def natural_key(path: str):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_raw_files(in_dir: str):
    files = []
    for name in os.listdir(in_dir):
        if name.lower().endswith(".raw"):
            files.append(os.path.join(in_dir, name))
    return sorted(files, key=natural_key)


def load_raw(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype=DTYPE)
    expected = IMG_H * IMG_W
    if arr.size != expected:
        raise ValueError(f"Size mismatch: {path}: got={arr.size}, expected={expected}")
    return arr.reshape((IMG_H, IMG_W))


def norm_for_view(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def pick_sample_files(files):
    if len(files) <= 3:
        return files
    return [files[0], files[len(files) // 2], files[-1]]


def save_preview(full_img, crop_img, out_path, title):
    full01 = norm_for_view(full_img)
    crop01 = norm_for_view(crop_img)

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(full01, cmap="gray")
    ax1.add_patch(Rectangle((X0, Y0), W, H, fill=False, linewidth=1.5))
    ax1.set_title("full image + ROI")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(crop01, cmap="gray")
    ax2.set_title(f"crop H={H}, W={W}")
    ax2.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    if not (0 <= X0 < X1 <= IMG_W and 0 <= Y0 < Y1 <= IMG_H):
        raise ValueError(f"ROI out of bounds: x[{X0},{X1}) y[{Y0},{Y1})")

    os.makedirs(OUT_DIR, exist_ok=True)

    for g in GROUPS_FOR_PREVIEW:
        in_dir = os.path.join(ROOT, g)
        if not os.path.isdir(in_dir):
            print(f"[SKIP] missing dir: {in_dir}")
            continue

        files = list_raw_files(in_dir)
        if not files:
            print(f"[WARN] no RAW files in {in_dir}")
            continue

        samples = pick_sample_files(files)

        for p in samples:
            img = load_raw(p)
            crop = img[Y0:Y1, X0:X1]
            base = os.path.splitext(os.path.basename(p))[0]
            out_png = os.path.join(OUT_DIR, f"{g}_{base}_x{X0}_y{Y0}_w{W}_h{H}.png")
            save_preview(img, crop, out_png, title=f"{g}/{os.path.basename(p)}")
            print(f"[OK] saved {out_png}")

    print("Done.")
    print(f"Check previews in: {OUT_DIR}")


if __name__ == "__main__":
    main()
