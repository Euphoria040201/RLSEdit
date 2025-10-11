import argparse
import json
import os
import re
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_mean(x):
    return float(np.mean(x)) if len(x) else np.nan


def _extract_edits_from_path(p: Path) -> int | None:
    m = re.search(r"edits_(\d+)", p.as_posix())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m = re.search(r"forgetting_report_at_(\d+)_edits\.json", p.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    try:
        with open(p, "r") as f:
            j = json.load(f)
        v = int(j.get("checkpoint_total_edits", 0))
        return v if v > 0 else None
    except Exception:
        return None


def _find_forgetting_reports(run_dir: Path, ds_name: str | None):

    candidates: list[Path] = []


    if ds_name:
        candidates += list(run_dir.glob(f"eval/edits_*/{ds_name}/forgetting_report.json"))
    else:
        candidates += list(run_dir.glob(f"eval/edits_*/**/forgetting_report.json"))

  
    candidates += list(run_dir.glob("forgetting_report_at_*_edits.json"))

    for p in run_dir.rglob("forgetting_report*.json"):
        if p not in candidates:
            candidates.append(p)


    by_E: dict[int, Path] = {}
    priority_score = lambda path: 0 if "/eval/" in path.as_posix() else 1

    for p in candidates:
        E = _extract_edits_from_path(p)
        if E is None:
            continue
        if E not in by_E or priority_score(p) < priority_score(by_E[E]):
            by_E[E] = p

    if not by_E:
        raise FileNotFoundError(f"No forgetting reports found under: {run_dir}")

    return dict(sorted(by_E.items(), key=lambda kv: kv[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ds_name", default="mcf")
    ap.add_argument("--out_subdir")
    ap.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.0), help="y 轴范围，默认 0~1")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    assert run_dir.exists(), f"run_dir 不存在：{run_dir}"

    ds_name = args.ds_name.strip() if args.ds_name else None
    reports = _find_forgetting_reports(run_dir, ds_name)
    checkpoints = list(reports.keys())
    print(f"[plot] found {len(checkpoints)} checkpoints: {checkpoints[:6]}{' ...' if len(checkpoints)>6 else ''}")

    # 读取所有报告 -> 三个字典 {E: {case_id: metric}}
    R, P, N = {}, {}, {}
    sizes_by_E = {}  # 记录每个 E 的 detailed_results 样本数
    for E, fp in reports.items():
        with open(fp, "r") as f:
            data = json.load(f)
        r, p, n = {}, {}, {}
        for item in data.get("detailed_results", []):
            cid = int(item.get("case_id", -1))
            if cid < 0:
                continue
            r[cid] = float(item.get("rewrite_acc", item.get("accuracy", np.nan)))
            p[cid] = float(item.get("paraphrase_acc", np.nan))
            n[cid] = float(item.get("neighborhood_acc", np.nan))
        R[E], P[E], N[E] = r, p, n
        sizes_by_E[E] = len(r)  
    cum_sets = OrderedDict((E, set(R[E].keys())) for E in checkpoints)

    groups = OrderedDict()
    prev = set()
    for E in checkpoints:
        groups[E] = sorted(list(cum_sets[E] - prev))
        prev = cum_sets[E]

    if args.out_subdir:
        out_dir = run_dir / args.out_subdir
    else:
        out_dir = run_dir / "eval" / "plots" / (ds_name if ds_name else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot] outputs -> {out_dir}")


    xs, y_r, y_p, y_n = [], [], [], []
    ns = []
    for E in checkpoints:
        xs.append(E)
        rs = list(R[E].values())
        ps = list(P[E].values())
        ns_ = list(N[E].values())
        y_r.append(_safe_mean(rs))
        y_p.append(_safe_mean(ps))
        y_n.append(_safe_mean(ns_))
        ns.append(len(rs))

    plt.figure()
    plt.plot(xs, y_r, label="rewrite")
    plt.plot(xs, y_p, label="paraphrase")
    plt.plot(xs, y_n, label="neighborhood")
    plt.title("Overall average accuracy vs. edits")
    plt.xlabel("Checkpoint total edits")
    plt.ylabel("Average accuracy (all cases up to that checkpoint)")
    plt.ylim(*args.ylim)
    plt.xticks(xs, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "overview.png")
    plt.close()

    pd.DataFrame({
        "checkpoint_total_edits": xs,
        "avg_rewrite": y_r,
        "avg_paraphrase": y_p,
        "avg_neighborhood": y_n,
        "num_cases": ns,
    }).to_csv(out_dir / "overview.csv", index=False)

    all_group_frames = []
    for E_def, group_cases in groups.items():
        if not group_cases:
            continue
        xs_g, yr, yp, yn = [], [], [], []
        nr, np_, nn = [], [], []

        for E in checkpoints:
            if E < E_def:
                continue

            rw = [R[E][c] for c in group_cases if c in R[E]]
            pp = [P[E][c] for c in group_cases if c in P[E]]
            nn_ = [N[E][c] for c in group_cases if c in N[E]]
            if not (rw or pp or nn_):
                continue
            xs_g.append(E)
            yr.append(_safe_mean(rw))
            yp.append(_safe_mean(pp))
            yn.append(_safe_mean(nn_))
            nr.append(len(rw))
            np_.append(len(pp))
            nn.append(len(nn_))

        if not xs_g:
            continue

        gmin, gmax = min(group_cases), max(group_cases)
        plt.figure()
        plt.plot(xs_g, yr, label="rewrite")
        plt.plot(xs_g, yp, label="paraphrase")
        plt.plot(xs_g, yn, label="neighborhood")
        plt.title(f"Group defined at {E_def} edits (cases ~{gmin}-{gmax}, size={len(group_cases)})")
        plt.xlabel("Checkpoint total edits")
        plt.ylabel("Average accuracy over group")
        plt.ylim(*args.ylim)
        plt.xticks(xs_g, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"group_{E_def:06d}.png")
        plt.close()

        df = pd.DataFrame({
            "group_defined_at_edits":        [E_def]*len(xs_g),
            "checkpoint_total_edits":        xs_g,
            "avg_rewrite_accuracy":          yr,
            "num_rewrite_cases_present":     nr,
            "avg_paraphrase_accuracy":       yp,
            "num_paraphrase_cases_present":  np_,
            "avg_neighborhood_accuracy":     yn,
            "num_neighborhood_cases_present":nn,
            "group_case_min_id":             [gmin]*len(xs_g),
            "group_case_max_id":             [gmax]*len(xs_g),
            "num_cases_in_group":            [len(group_cases)]*len(xs_g),
        })
        df.to_csv(out_dir / f"group_{E_def:06d}.csv", index=False)
        all_group_frames.append(df)

    if all_group_frames:
        big = pd.concat(all_group_frames, ignore_index=True)
        big.to_csv(out_dir / "all_groups_summary.csv", index=False)

    print("[plot] done.")


if __name__ == "__main__":
    main()
