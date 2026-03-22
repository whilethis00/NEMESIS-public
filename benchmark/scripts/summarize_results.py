#!/usr/bin/env python3
"""
Collect all experiment results and print/save summary tables.

Usage:
    python summarize_results.py --results_dir results/

Output:
    - Console: formatted tables (Dice, HD95)
    - results/summary_synapse.csv
    - results/summary_label_eff.csv
    - results/summary_ood.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SYNAPSE_ORGANS = [
    "aorta", "gallbladder", "spleen",
    "left_kidney", "right_kidney", "liver", "stomach", "pancreas",
]


def load_results(results_dir: Path, exp_name: str) -> dict | None:
    """Load test_results.json from experiment directory."""
    candidates = [
        results_dir / exp_name / "test_results.json",
        results_dir / exp_name / "metrics.csv",  # fallback: last row of CSV
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".json":
                return json.loads(p.read_text())
            # CSV: return last row as dict
            rows = list(csv.DictReader(p.read_text().splitlines()))
            if rows:
                return rows[-1]
    return None


def print_synapse_table(results_dir: Path):
    experiments = {
        "NEMESIS (ours)":  "synapse_nemesis_full",
        "Random ViT":      "synapse_random_vit",
        "SwinUNETR":       "synapse_swinunetr",
    }

    print("\n" + "=" * 90)
    print("Table: Synapse Multi-organ Segmentation (Dice % / HD95 mm)")
    print("=" * 90)

    header = f"{'Method':<22}" + "".join(f"{o[:8]:>10}" for o in SYNAPSE_ORGANS) + f"{'Mean DSC':>10}{'Mean HD95':>11}"
    print(header)
    print("-" * 90)

    rows_csv = []
    for method, exp in experiments.items():
        r = load_results(results_dir, exp)
        if r is None:
            print(f"  {method:<22} -- (not found)")
            continue

        if "per_class" in r:
            dices = [r["per_class"].get(o, {}).get("dice", 0.0) * 100 for o in SYNAPSE_ORGANS]
            hd95s = [r["per_class"].get(o, {}).get("hd95", float("nan")) for o in SYNAPSE_ORGANS]
            mean_d = r.get("mean_dice", sum(dices)/len(dices)/100) * 100
            mean_h = r.get("mean_hd95", 0.0)
        else:
            # CSV row
            dices = [float(r.get(f"dice_{o}", 0)) * 100 for o in SYNAPSE_ORGANS]
            hd95s = [float("nan")] * len(SYNAPSE_ORGANS)
            mean_d = float(r.get("val_mean_dice", 0)) * 100
            mean_h = float(r.get("val_mean_hd95", 0))

        row_str = f"  {method:<22}" + "".join(f"{d:>9.2f}%" for d in dices) + f"{mean_d:>9.2f}%{mean_h:>10.2f}"
        print(row_str)
        rows_csv.append([method] + [f"{d:.2f}" for d in dices] + [f"{mean_d:.2f}", f"{mean_h:.2f}"])

    print("=" * 90)

    # Save CSV
    out_csv = results_dir / "summary_synapse.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + SYNAPSE_ORGANS + ["Mean_Dice%", "Mean_HD95mm"])
        w.writerows(rows_csv)
    print(f"Saved: {out_csv}")


def print_label_eff_table(results_dir: Path):
    fracs  = ["1", "5", "10", "100"]
    models = ["nemesis", "random_vit", "swinunetr"]
    labels = {"nemesis": "NEMESIS", "random_vit": "Random ViT", "swinunetr": "SwinUNETR"}

    print("\n" + "=" * 60)
    print("Table: Label Efficiency (Mean Dice %)")
    print("=" * 60)

    header = f"{'Method':<16}" + "".join(f"{''+pct+'%':>10}" for pct in fracs)
    print(header)
    print("-" * 60)

    rows_csv = []
    for m in models:
        dices = []
        for pct in fracs:
            r = load_results(results_dir / "label_eff", f"{m}_{pct}pct")
            if r is None:
                dices.append("--")
            else:
                if "mean_dice" in r:
                    dices.append(f"{float(r['mean_dice'])*100:.2f}")
                else:
                    dices.append(f"{float(r.get('val_mean_dice', 0))*100:.2f}")
        row = f"  {labels[m]:<14}" + "".join(f"{d:>10}" for d in dices)
        print(row)
        rows_csv.append([labels[m]] + dices)

    print("=" * 60)

    out_csv = results_dir / "summary_label_eff.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + [f"{p}%" for p in fracs])
        w.writerows(rows_csv)
    print(f"Saved: {out_csv}")


def print_ood_table(results_dir: Path):
    datasets = {"KiTS23": "kits23", "MSD Pancreas": "msd_pancreas"}
    models   = {
        "NEMESIS":     "{ds}_nemesis",
        "Random ViT":  "{ds}_random_vit",
        "SwinUNETR":   "{ds}_swinunetr",
    }

    print("\n" + "=" * 55)
    print("Table: OOD Generalization (Mean Dice %)")
    print("=" * 55)
    header = f"{'Method':<16}" + "".join(f"{n:>18}" for n in datasets)
    print(header)
    print("-" * 55)

    rows_csv = []
    for m_label, m_tmpl in models.items():
        dices = []
        for ds_name, ds_key in datasets.items():
            exp = m_tmpl.format(ds=ds_key)
            r   = load_results(results_dir, exp)
            if r is None:
                dices.append("--")
            else:
                v = r.get("mean_dice", r.get("val_mean_dice", 0))
                dices.append(f"{float(v)*100:.2f}")
        print(f"  {m_label:<14}" + "".join(f"{d:>18}" for d in dices))
        rows_csv.append([m_label] + dices)

    print("=" * 55)

    out_csv = results_dir / "summary_ood.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + list(datasets))
        w.writerows(rows_csv)
    print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print_synapse_table(results_dir)
    print_label_eff_table(results_dir)
    print_ood_table(results_dir)


if __name__ == "__main__":
    main()
