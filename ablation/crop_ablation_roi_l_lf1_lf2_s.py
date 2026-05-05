import os
import re
import numpy as np

# ============================================================
# Ablation ROI crop script
# Groups:
#   l   : large focus, no speckle
#   lf1 : large focus + speckle, position 1
#   lf2 : large focus + speckle, position 2
#   s   : small focus
#
# Important:
#   1. Decide ROI by checking lf1/lf2, because they contain the 3D printed frame.
#   2. Apply exactly the same ROI to l, lf1, lf2, and s.
#   3. Do not resize. Only crop.
# ============================================================

IN_ROOT  = r"D:\research\niboshi"
OUT_ROOT = r"D:\research\niboshi\roi_ablation"

GROUPS = ["l", "lf1", "lf2", "s"]

# Original RAW image size before cropping
IMG_H, IMG_W = 2944, 2352
DTYPE = np.uint16

# ROI format: x from X0 to X0 + W, y from Y0 to Y0 + H
X0, Y0 = 56, 478
W,  H  = 1984, 1856
X1, Y1 = X0 + W, Y0 + H


def natural_key(path: str):
    """Sort 00000.RAW, 00001.RAW, ..., 00468.RAW in numeric order."""
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
        raise ValueError(
            f"Size mismatch: {path}\n"
            f"  got={arr.size}, expected={expected} for H={IMG_H}, W={IMG_W}"
        )
    return arr.reshape((IMG_H, IMG_W))


def save_raw(path: str, img: np.ndarray) -> None:
    img.astype(DTYPE).tofile(path)


def main():
    if not (0 <= X0 < X1 <= IMG_W and 0 <= Y0 < Y1 <= IMG_H):
        raise ValueError(
            f"ROI out of bounds: x[{X0},{X1}) y[{Y0},{Y1}) "
            f"for original H={IMG_H}, W={IMG_W}"
        )

    os.makedirs(OUT_ROOT, exist_ok=True)

    counts = {}

    for g in GROUPS:
        in_dir  = os.path.join(IN_ROOT, g)
        out_dir = os.path.join(OUT_ROOT, g)

        if not os.path.isdir(in_dir):
            print(f"[SKIP] missing dir: {in_dir}")
            continue

        files = list_raw_files(in_dir)
        if not files:
            print(f"[WARN] no RAW files in {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        for i, f in enumerate(files):
            img = load_raw(f)
            crop = img[Y0:Y1, X0:X1]
            out_path = os.path.join(out_dir, os.path.basename(f))
            save_raw(out_path, crop)

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"[{g}] {i + 1}/{len(files)}")

        counts[g] = len(files)
        print(f"[OK] {g}: {len(files)} files -> cropped to H={H}, W={W}")

    if counts:
        print("\nCounts:")
        for g, c in counts.items():
            print(f"  {g}: {c}")

        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            print("[WARN] group file counts are not identical. Check missing frames before pairing.")

    print("\nDone.")
    print(f"Cropped root: {OUT_ROOT}")
    print(f"Use cropped size in later scripts: h={H}, w={W}")


if __name__ == "__main__":
    main()
