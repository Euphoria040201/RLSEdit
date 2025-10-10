
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math

import matplotlib.pyplot as plt

LAYER_RE = re.compile(r'^\s*LAYER\s+(\d+)\s*$', re.IGNORECASE)
UPD_NORM_RE = re.compile(r'upd norm tensor\(\s*([-+0-9.eE]+)')

def parse_log_one(path: Path) -> Dict[int, List[float]]:
    per_layer: Dict[int, List[float]] = {}
    cur_layer: int = None
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            mL = LAYER_RE.search(line)
            if mL:
                cur_layer = int(mL.group(1))
                per_layer.setdefault(cur_layer, [])
                continue
            if cur_layer is not None:
                mU = UPD_NORM_RE.search(line)
                if mU:
                    try:
                        val = float(mU.group(1))
                        per_layer[cur_layer].append(val)
                    except ValueError:
                        pass
    return {k: v for k, v in per_layer.items() if len(v) > 0}

def align_series(a: List[float], b: List[float]) -> Tuple[List[float], List[float]]:
    n = max(len(a), len(b))
    a2 = list(a) + [float('nan')] * (n - len(a))
    b2 = list(b) + [float('nan')] * (n - len(b))
    return a2, b2

def nanmean(xs: List[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return sum(vals)/len(vals) if vals else float('nan')

def nanmedian(xs: List[float]) -> float:
    vals = sorted([x for x in xs if not math.isnan(x)])
    if not vals:
        return float('nan')
    m = len(vals)//2
    if len(vals) % 2 == 1:
        return vals[m]
    return 0.5*(vals[m-1] + vals[m])

def nanmax(xs: List[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return max(vals) if vals else float('nan')

def nan_auc(xs: List[float]) -> float:
    # simple sum as proxy for area (uniform step = 1)
    vals = [x for x in xs if not math.isnan(x)]
    return sum(vals) if vals else float('nan')

def plot_overlay(layer: int, a: List[float], b: List[float], title: str, out_png: Path, dpi=150):
    plt.figure(figsize=(8, 5))
    x = list(range(1, max(len(a), len(b)) + 1))
    a2, b2 = align_series(a, b)
    plt.plot(x, a2, label="Run A", linewidth=1.5)
    plt.plot(x, b2, label="Run B", linewidth=1.5)
    plt.xlabel("Edit Index (per layer)")
    plt.ylabel("upd norm")
    plt.title(f"{title} — Layer {layer}")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

def plot_diff(layer: int, a: List[float], b: List[float], title: str, out_png: Path, dpi=150):
    plt.figure(figsize=(8, 5))
    a2, b2 = align_series(a, b)
    # diff: B - A
    diff = []
    for x, y in zip(b2, a2):
        if math.isnan(x) or math.isnan(y):
            diff.append(float('nan'))
        else:
            diff.append(x - y)
    x = list(range(1, len(diff) + 1))
    plt.plot(x, diff, label="(Run B - Run A)", linewidth=1.5)
    plt.xlabel("Edit Index (per layer)")
    plt.ylabel("Δ upd norm")
    plt.title(f"{title} — Layer {layer}")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

def save_timeseries_csv(layer: int, a: List[float], b: List[float], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    a2, b2 = align_series(a, b)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["edit_idx", "runA_upd_norm", "runB_upd_norm", "diff_B_minus_A"])
        for i, (x, y) in enumerate(zip(a2, b2)):
            if math.isnan(x) or math.isnan(y):
                d = ""
            else:
                d = y - x
            w.writerow([i+1, "" if math.isnan(x) else x, "" if math.isnan(y) else y, d])

def save_summary_csv(summary_rows: List[List], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hdr = [
        "layer",
        "runA_mean", "runA_median", "runA_max", "runA_auc",
        "runB_mean", "runB_median", "runB_max", "runB_auc",
        "diff_mean(B-A)", "diff_median(B-A)", "diff_max(B-A)", "diff_auc(B-A)"
    ]
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(summary_rows)

def main():
    ap = argparse.ArgumentParser(description="Compare 'upd norm' per layer between two logs.")
    ap.add_argument("runA", help="Path to log A")
    ap.add_argument("runB", help="Path to log B")
    ap.add_argument("-o", "--outdir", default="upd_norm_compare", help="Output directory")
    ap.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    ap.add_argument("--perlayer", action="store_true", help="Export per-layer CSV time series")
    args = ap.parse_args()

    pA = Path(args.runA)
    pB = Path(args.runB)
    if not pA.exists() or not pB.exists():
        print("[ERR] One or both input paths do not exist.")
        return

    dataA = parse_log_one(pA)
    dataB = parse_log_one(pB)
    layers = sorted(set(dataA.keys()) | set(dataB.keys()))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    title = f"upd norm — {pA.stem} vs {pB.stem}"

    summary_rows: List[List] = []
    for L in layers:
        a = dataA.get(L, [])
        b = dataB.get(L, [])
        # plots
        plot_overlay(L, a, b, title, outdir / f"overlay_layer_{L}.png", dpi=args.dpi)
        plot_diff(L, a, b, title, outdir / f"diff_layer_{L}.png", dpi=args.dpi)
        # per-layer timeseries csv
        if args.perlayer:
            save_timeseries_csv(L, a, b, outdir / f"timeseries_layer_{L}.csv")
        # summary stats
        a2, b2 = align_series(a, b)
        # compute diff B-A elementwise ignoring NaNs
        diff = []
        for x, y in zip(a2, b2):
            if math.isnan(x) or math.isnan(y):
                continue
            diff.append(y - x)
        row = [
            L,
            nanmean(a2), nanmedian(a2), nanmax(a2), nan_auc(a2),
            nanmean(b2), nanmedian(b2), nanmax(b2), nan_auc(b2),
            nanmean(diff), nanmedian(diff), nanmax(diff) if diff else float('nan'), nan_auc(diff)
        ]
        summary_rows.append(row)

    # save summary
    save_summary_csv(summary_rows, outdir / "summary.csv")

if __name__ == "__main__":
    main()
