# experiments/eval_run_checkpoints.py
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from glue_eval.glue_eval import GLUEEval
def _extract_edits_num(p: Path) -> Optional[int]:
    m = re.search(r"edits_(\d+)", p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _find_all_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    cands = []
    for p in run_dir.rglob("edits_*"):
        if not p.is_dir():
            continue
        if not (p / "config.json").exists():
            continue
        n = _extract_edits_num(p)
        if n is None:
            continue
        cands.append((n, p))
    cands.sort(key=lambda x: x[0])
    return cands

def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "experiments").is_dir() and (p / "experiments" / "eval_checkpoint_single.py").exists():
            return p
    return start

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="形如 results/AlphaEdit/run_090 的目录")
    ap.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre", "mquake"])
    ap.add_argument("--dataset_size_limit", type=int, default=None)
    ap.add_argument("--generation_test_interval", type=int, default=5,
                    help="-1 仅快评测；>=1 表示每 N 条做一次生成评测")
    ap.add_argument("--trust_remote_code", action="store_true", help="Qwen 等需要 True；Llama 不需要")
    ap.add_argument("--min_edits", type=int, default=None, help="仅评测 edits >= 此值 的 checkpoint")
    ap.add_argument("--max_edits", type=int, default=None, help="仅评测 edits <= 此值 的 checkpoint")
    ap.add_argument("--skip_existing", action="store_true", help="若对应输出目录已存在 summary.json 则跳过")
    ap.add_argument("--overwrite", action="store_true", help="传给子进程，让其覆盖 case_*.json")
    ap.add_argument("--dry_run", action="store_true", help="只打印将要执行的命令，不实际运行")
    ap.add_argument("--python_exec", default="python3", help="指定 python 可执行文件（默认 python3）")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    assert run_dir.exists(), f"run_dir 不存在：{run_dir}"

    repo_root = _find_repo_root(run_dir)
    print(f"[eval-run] repo_root = {repo_root}")

    ckpts = _find_all_checkpoints(run_dir)
    if not ckpts:
        print(f"[eval-run] 未在 {run_dir} 下找到任何 edits_* checkpoint")
        return

    logs_dir = run_dir / "eval" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for n_edits, ckpt in ckpts:
        if args.min_edits is not None and n_edits < args.min_edits:
            continue
        if args.max_edits is not None and n_edits > args.max_edits:
            continue

        out_dir = run_dir / "eval" / ckpt.name / args.ds_name
        summary_json = out_dir / "summary.json"
        if args.skip_existing and summary_json.exists():
            print(f"[eval-run] 跳过 {ckpt}（已存在 {summary_json}）")
            continue

        cmd = [
            args.python_exec, "-u", "-m", "experiments.eval_checkpoint_single",
            "--ckpt_dir", str(ckpt),
            "--ds_name", args.ds_name,
            "--generation_test_interval", str(args.generation_test_interval),
            "--to_run_dir",
            "--infer_from_ckpt_meta",
        ]
        if args.dataset_size_limit is not None:
            cmd += ["--dataset_size_limit", str(args.dataset_size_limit)]
        if args.trust_remote_code:
            cmd += ["--trust_remote_code"]
        if args.overwrite:
            cmd += ["--overwrite"]

        log_file = logs_dir / f"edits_{n_edits:06d}.log"
        print(f"[eval-run] ({n_edits}) {ckpt}")
        print("          >", " ".join(cmd))
        if args.dry_run:
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{str(repo_root)}{os.pathsep}{env.get('PYTHONPATH', '')}"

        with open(log_file, "w") as lf:
            lf.write(f"# CKPT: {ckpt}\n")
            lf.write(f"# CMD : {' '.join(cmd)}\n\n")
            lf.flush()
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=repo_root,
                env=env,
                check=False,
            )
        if proc.returncode == 0:
            print(f"[eval-run] 完成 {ckpt} -> {summary_json}")
        else:
            print(f"[eval-run][ERR] {ckpt} 评测子进程退出码={proc.returncode}，详见 {log_file}")

    print("[eval-run] 全部完成。")

if __name__ == "__main__":
    main()
