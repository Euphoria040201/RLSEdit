# repair_k0.py
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

DTYPE_MAP = {
    "float32": np.float32,
    "fp32": np.float32,
    "f32": np.float32,
    "float16": np.float16,
    "fp16": np.float16,
    "f16": np.float16,
}

def infer_shape(file_size_bytes: int, d: int, dtype: np.dtype) -> Tuple[int, int]:
    itemsize = np.dtype(dtype).itemsize
    if file_size_bytes % itemsize != 0:
        raise ValueError(f"file size {file_size_bytes} not multiple of dtype {dtype} itemsize {itemsize}")
    elems = file_size_bytes // itemsize
    if elems % d != 0:
        raise ValueError(f"total element count {elems} not divisible by d={d}")
    N = elems // d
    return d, N

def raw_to_ndarray(path_raw: str, d: int, dtype: np.dtype, transpose: bool) -> np.ndarray:
    sz = os.path.getsize(path_raw)
    d, N = infer_shape(sz, d, dtype)
    print(f"[INFO] {path_raw}: infer shape=({d}, {N}), dtype={np.dtype(dtype).name}")

    # (d, N) row-major
    mm = np.memmap(path_raw, mode="r", dtype=dtype, shape=(d, N))
    arr = np.array(mm, copy=True)  # decouple from underlying file
    del mm

    if transpose:
        arr = arr.T  # -> (N, d) if你需要反过来存
        print(f"[INFO] transposed -> shape={arr.shape}")

    # 轻量统计
    try:
        sample = arr.reshape(-1)
        k = min(sample.size, 100000)
        s = sample[:k]
        print(f"[STAT] min={float(s.min()):.4g} max={float(s.max()):.4g} mean={float(s.mean()):.4g} "
              f"std={float(s.std()):.4g} frob≈{float(np.linalg.norm(s)): .4g} (sample {k})")
        if not np.isfinite(s).all():
            print("[WARN] sample contains non-finite values (nan/inf).")
    except Exception as e:
        print(f"[WARN] stats failed: {e}")

    # 一律保存为 float32（可选：保持原 dtype）
    if dtype == np.float16:
        arr = arr.astype(np.float32, copy=False)

    return arr

def write_npy_safely(out_path: str, arr: np.ndarray, backup: bool):
    out_path = str(out_path)
    if backup and os.path.exists(out_path):
        bak = out_path + ".bak"
        if not os.path.exists(bak):
            os.link(out_path, bak)  # 硬链接备份
            print(f"[BACKUP] -> {bak}")

    tmp = out_path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, out_path)
    print(f"[OK] wrote npy -> {out_path} (shape={arr.shape}, dtype={arr.dtype})")

def iter_targets(paths: List[str], pattern: Optional[str]) -> List[str]:
    out = []
    for p in paths:
        P = Path(p)
        if P.is_dir():
            if pattern is None:
                # 默认找所有文件
                out += [str(f) for f in P.rglob("*") if f.is_file()]
            else:
                out += [str(f) for f in P.rglob(pattern) if f.is_file()]
        elif P.is_file():
            out.append(str(P))
        else:
            print(f"[SKIP] not found: {p}")
    return out

def main():
    ap = argparse.ArgumentParser(description="Repair raw K0 binary into .npy (shape (d, N)).")
    ap.add_argument("paths", nargs="+", help="文件或目录（可多个）")
    ap.add_argument("--d", type=int, default=6400, help="每列维度 d（默认 6400，GPT-2-XL c_proj 输入维度）")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP.keys()),
                    help="原始裸文件元素类型（默认 float32）")
    ap.add_argument("--transpose", action="store_true",
                    help="写入前转置（把 (d,N) 变为 (N,d)）")
    ap.add_argument("--pattern", default=None,
                    help="当输入是目录时的文件匹配（如 '*.raw' 或 '*_K.npy'）。默认不过滤。")
    ap.add_argument("--suffix", default=".npy",
                    help="输出文件后缀（默认 .npy，若传入的原文件名已以 .npy 结尾会直接覆盖）")
    ap.add_argument("--outdir", default=None, help="输出目录。默认就地覆盖/写入。")
    ap.add_argument("--fix", action="store_true", help="执行写入；否则仅 dry-run 预览。")
    ap.add_argument("--backup", action="store_true", help="覆盖前创建 .bak 硬链接备份（仅 --fix 时生效）。")
    args = ap.parse_args()

    dtype = DTYPE_MAP[args.dtype.lower()]
    files = iter_targets(args.paths, args.pattern)
    if not files:
        print("[ERROR] no files matched.")
        sys.exit(1)

    print(f"[INFO] matched {len(files)} file(s). d={args.d}, dtype={np.dtype(dtype).name}, transpose={args.transpose}")
    for src in sorted(files):
        try:
            arr = raw_to_ndarray(src, args.d, dtype, args.transpose)

            if args.outdir:
                Path(args.outdir).mkdir(parents=True, exist_ok=True)
                base = Path(src).name
                if args.suffix and not base.endswith(args.suffix):
                    base = base + args.suffix
                dst = str(Path(args.outdir) / base)
            else:
                # 就地：若不是 .npy，自动加 .npy；是 .npy 则覆盖
                if src.endswith(".npy"):
                    dst = src
                else:
                    dst = src + args.suffix

            if args.fix:
                write_npy_safely(dst, arr, backup=args.backup)
            else:
                print(f"[DRY-RUN] would write -> {dst} (shape={arr.shape}, dtype={arr.dtype})")
        except Exception as e:
            print(f"[ERROR] {src}: {e}")

if __name__ == "__main__":
    main()
