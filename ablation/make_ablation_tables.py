import os
import csv
import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# =========================================================
# 0) CONFIG
# =========================================================
TRAIN_RUNS_ROOT = r"D:\research\niboshi\train_runs"

# Add or remove experiments here.
# The script will skip experiments whose summary.csv does not exist.
EXPERIMENTS = [
    {
        "key": "l",
        "label": "L → S",
        "dir": "l_to_s_train3",
    },
    {
        "key": "lf1",
        "label": "LF1 → S",
        "dir": "lf1_to_s_train3",
    },
    {
        "key": "lf1_lf1",
        "label": "LF1 + LF1 → S",
        "dir": "lf1_lf1_to_s_train3",
    },
    {
        "key": "lf1_lf2",
        "label": "LF1 + LF2 → S",
        "dir": "lf1_lf2_to_s_train3",
    },
]
PRED_DIR_NAME = "predict_best"

OUT_DIR = os.path.join(TRAIN_RUNS_ROOT, "_ablation_tables")
DPI = 300


# =========================================================
# 1) IO
# =========================================================
def read_summary_csv(path: str) -> List[Dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"index", "name", "masked_psnr", "masked_mae"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"summary.csv must contain {required}, got {reader.fieldnames}")

        for r in reader:
            rows.append({
                "index": int(r["index"]),
                "name": r["name"],
                "masked_psnr": float(r["masked_psnr"]),
                "masked_mae": float(r["masked_mae"]),
            })
    return rows


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def std_sample(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x: Optional[float], ndigits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{ndigits}f}"


# =========================================================
# 2) TABLE IMAGE HELPERS
# =========================================================
def save_table_png(
    out_png: str,
    title: str,
    col_labels: List[str],
    cell_text: List[List[str]],
    fig_width: float = 10.5,
    row_height: float = 0.45,
) -> None:
    n_rows = len(cell_text)
    fig_height = max(2.2, 1.0 + row_height * (n_rows + 1))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_bar_png(out_png: str, labels: List[str], psnrs: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, psnrs)
    ax.set_ylabel("Mean masked PSNR (dB)")
    ax.set_title("Ablation Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, v in enumerate(psnrs):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 3) MAIN
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    loaded = []
    for exp in EXPERIMENTS:
        summary_path = os.path.join(
            TRAIN_RUNS_ROOT,
            exp["dir"],
            PRED_DIR_NAME,
            "summary.csv",
        )

        if not os.path.exists(summary_path):
            print(f"[SKIP] summary not found: {summary_path}")
            continue

        rows = read_summary_csv(summary_path)
        if not rows:
            print(f"[SKIP] empty summary: {summary_path}")
            continue

        loaded.append({
            **exp,
            "summary_path": summary_path,
            "rows": rows,
        })
        print(f"[OK] loaded {exp['label']}: {summary_path}")

    if not loaded:
        raise FileNotFoundError("No summary.csv files were found. Run predict_selected_angles.py first.")

    # ---------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------
    baseline = None
    for exp in loaded:
        if exp["key"] == "lf1":
            baseline = exp
            break

    baseline_mean_psnr = None
    baseline_mean_mae = None
    if baseline is not None:
        baseline_mean_psnr = mean([r["masked_psnr"] for r in baseline["rows"]])
        baseline_mean_mae = mean([r["masked_mae"] for r in baseline["rows"]])

    summary_rows = []
    for exp in loaded:
        psnrs = [r["masked_psnr"] for r in exp["rows"]]
        maes = [r["masked_mae"] for r in exp["rows"]]
        m_psnr = mean(psnrs)
        s_psnr = std_sample(psnrs)
        m_mae = mean(maes)
        s_mae = std_sample(maes)

        delta_psnr = None
        delta_mae = None
        if baseline_mean_psnr is not None:
            delta_psnr = m_psnr - baseline_mean_psnr
        if baseline_mean_mae is not None:
            delta_mae = m_mae - baseline_mean_mae

        summary_rows.append({
            "Experiment": exp["label"],
            "N": len(exp["rows"]),
            "Mean masked PSNR (dB)": m_psnr,
            "Std PSNR": s_psnr,
            "Delta PSNR vs LF1 (dB)": delta_psnr,
            "Mean masked MAE": m_mae,
            "Std MAE": s_mae,
            "Delta MAE vs LF1": delta_mae,
        })

    summary_csv = os.path.join(OUT_DIR, "ablation_summary_table.csv")
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    summary_table_text = []
    for r in summary_rows:
        summary_table_text.append([
            r["Experiment"],
            str(r["N"]),
            fmt(r["Mean masked PSNR (dB)"], 3),
            fmt(r["Std PSNR"], 3),
            fmt(r["Delta PSNR vs LF1 (dB)"], 3),
            fmt(r["Mean masked MAE"], 5),
            fmt(r["Delta MAE vs LF1"], 5),
        ])

    summary_png = os.path.join(OUT_DIR, "ablation_summary_table.png")
    save_table_png(
        summary_png,
        "Ablation Summary",
        [
            "Experiment",
            "N",
            "Mean PSNR",
            "Std PSNR",
            "Delta PSNR vs LF1",
            "Mean MAE",
            "Delta MAE vs LF1",
        ],
        summary_table_text,
        fig_width=11.5,
    )

    # ---------------------------------------------------------
    # Per-angle table
    # ---------------------------------------------------------
    all_indices = sorted(set(
        r["index"]
        for exp in loaded
        for r in exp["rows"]
    ))

    exp_by_key = {
        exp["key"]: {r["index"]: r for r in exp["rows"]}
        for exp in loaded
    }
    label_by_key = {exp["key"]: exp["label"] for exp in loaded}

    per_angle_rows = []
    for idx in all_indices:
        row = {"index": idx}
        name = None
        for exp in loaded:
            rr = exp_by_key[exp["key"]].get(idx)
            if rr is not None:
                name = rr["name"]
                row[f"{exp['key']}_psnr"] = rr["masked_psnr"]
                row[f"{exp['key']}_mae"] = rr["masked_mae"]
            else:
                row[f"{exp['key']}_psnr"] = None
                row[f"{exp['key']}_mae"] = None

        row["name"] = name or f"{idx:05d}.RAW"

        if "lf1" in exp_by_key and "lf1_lf2" in exp_by_key:
            a = row.get("lf1_psnr")
            b = row.get("lf1_lf2_psnr")
            row["delta_lf1_lf2_vs_lf1_psnr"] = (b - a) if a is not None and b is not None else None

            a_mae = row.get("lf1_mae")
            b_mae = row.get("lf1_lf2_mae")
            row["delta_lf1_lf2_vs_lf1_mae"] = (b_mae - a_mae) if a_mae is not None and b_mae is not None else None

        per_angle_rows.append(row)

    per_angle_csv = os.path.join(OUT_DIR, "ablation_per_angle_table.csv")
    fieldnames = ["index", "name"]
    for exp in loaded:
        fieldnames += [f"{exp['key']}_psnr", f"{exp['key']}_mae"]
    fieldnames += ["delta_lf1_lf2_vs_lf1_psnr", "delta_lf1_lf2_vs_lf1_mae"]

    with open(per_angle_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in per_angle_rows:
            w.writerow(r)

    per_angle_text = []
    for r in per_angle_rows:
        line = [r["name"]]
        for exp in loaded:
            line.append(fmt(r.get(f"{exp['key']}_psnr"), 3))
        line.append(fmt(r.get("delta_lf1_lf2_vs_lf1_psnr"), 3))
        per_angle_text.append(line)

    per_angle_cols = ["Test view"] + [label_by_key[exp["key"]] for exp in loaded] + ["Delta LF1+LF2 - LF1"]

    per_angle_png = os.path.join(OUT_DIR, "ablation_per_angle_psnr_table.png")
    save_table_png(
        per_angle_png,
        "Per-view Masked PSNR Comparison",
        per_angle_cols,
        per_angle_text,
        fig_width=12.5,
        row_height=0.38,
    )

    bar_png = os.path.join(OUT_DIR, "ablation_mean_psnr_bar.png")
    save_bar_png(
        bar_png,
        [r["Experiment"] for r in summary_rows],
        [r["Mean masked PSNR (dB)"] for r in summary_rows],
    )

    print("\n[Saved]")
    print(summary_csv)
    print(summary_png)
    print(per_angle_csv)
    print(per_angle_png)
    print(bar_png)


if __name__ == "__main__":
    main()
