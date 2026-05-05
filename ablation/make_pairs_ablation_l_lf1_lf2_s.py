import os
import re
import csv

# ============================================================
# Make pair CSVs for ablation experiment.
#
# Default target / GT:
#   s = small focus
#
# Inputs:
#   l   -> s   : large focus without speckle baseline
#   lf1 -> s   : large focus + speckle position 1
#   lf2 -> s   : large focus + speckle position 2
#
# Output:
#   ROOT/pairs_l_to_s/pairs.csv
#   ROOT/pairs_lf1_to_s/pairs.csv
#   ROOT/pairs_lf2_to_s/pairs.csv
# ============================================================

ROOT = r"D:\research\niboshi\roi_ablation"

GT_GROUP = "s"
IN_GROUPS = ["l", "lf1", "lf2"]

PAIR_BY = "name"


def natural_key(path: str):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_raw_files(in_dir: str):
    files = []
    for name in os.listdir(in_dir):
        if name.lower().endswith(".raw"):
            files.append(os.path.join(in_dir, name))
    return sorted(files, key=natural_key)


def collect_raw_map(in_dir: str):
    """
    Return dict: lower-case basename -> original full path.
    This handles .RAW/.raw differences.
    """
    out = {}
    for p in list_raw_files(in_dir):
        key = os.path.basename(p).lower()
        out[key] = p
    return out


def existing_pair_paths_by_name(in_dir: str, gt_dir: str):
    in_map = collect_raw_map(in_dir)
    gt_map = collect_raw_map(gt_dir)

    common = sorted(set(in_map.keys()) & set(gt_map.keys()), key=natural_key)

    pairs = []
    for key in common:
        pairs.append((
            os.path.relpath(in_map[key], ROOT),
            os.path.relpath(gt_map[key], ROOT),
        ))

    missing_in = sorted(set(gt_map.keys()) - set(in_map.keys()), key=natural_key)
    missing_gt = sorted(set(in_map.keys()) - set(gt_map.keys()), key=natural_key)

    return pairs, missing_in, missing_gt


def existing_pair_paths_by_index(in_dir: str, gt_dir: str):
    in_files = list_raw_files(in_dir)
    gt_files = list_raw_files(gt_dir)

    n = min(len(in_files), len(gt_files))
    pairs = [
        (os.path.relpath(in_files[i], ROOT), os.path.relpath(gt_files[i], ROOT))
        for i in range(n)
    ]

    return pairs, len(in_files), len(gt_files)


def write_pairs_csv(out_csv: str, pairs):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["in_path", "gt_path"])
        for a, b in pairs:
            w.writerow([a, b])


def main():
    gt_dir = os.path.join(ROOT, GT_GROUP)
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    for g in IN_GROUPS:
        in_dir = os.path.join(ROOT, g)
        if not os.path.isdir(in_dir):
            print(f"[SKIP] missing dir: {in_dir}")
            continue

        if PAIR_BY == "name":
            pairs, missing_in, missing_gt = existing_pair_paths_by_name(in_dir, gt_dir)

            if missing_in:
                print(f"[WARN] {g}: {len(missing_in)} frames exist in GT but not in input. Example: {missing_in[:5]}")
            if missing_gt:
                print(f"[WARN] {g}: {len(missing_gt)} frames exist in input but not in GT. Example: {missing_gt[:5]}")

        elif PAIR_BY == "index":
            pairs, n_in, n_gt = existing_pair_paths_by_index(in_dir, gt_dir)
            if n_in != n_gt:
                print(f"[WARN] {g}: input count={n_in}, gt count={n_gt}. Paired only first {len(pairs)} by index.")
        else:
            raise ValueError(f"Unknown PAIR_BY: {PAIR_BY}")

        if not pairs:
            print(f"[WARN] no valid pairs for {g}")
            continue

        out_dir = os.path.join(ROOT, f"pairs_{g}_to_{GT_GROUP}")
        out_csv = os.path.join(out_dir, "pairs.csv")
        write_pairs_csv(out_csv, pairs)

        print(f"[OK] {g} -> {GT_GROUP}: wrote {len(pairs)} pairs -> {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
