# check_K0.py
import numpy as np, os, sys, argparse

LAYER_PATHS = {
    13: "data/stats/K0_exports/gpt2-xl/transformer_h_13_mlp_c_proj_K.npy",
    14: "data/stats/K0_exports/gpt2-xl/transformer_h_14_mlp_c_proj_K.npy",
    15: "data/stats/K0_exports/gpt2-xl/transformer_h_15_mlp_c_proj_K.npy",
    16: "data/stats/K0_exports/gpt2-xl/transformer_h_16_mlp_c_proj_K.npy",
    17: "data/stats/K0_exports/gpt2-xl/transformer_h_17_mlp_c_proj_K.npy",
}

def safe_load(path):
    try:
        return np.load(path, mmap_mode="r"), {"allow_pickle": False, "mmap": True}
    except ValueError as e:
        # 典型：Cannot load file containing pickled data when allow_pickle=False
        arr = np.load(path, allow_pickle=True)  # 不要mmap，object不支持
        return arr, {"allow_pickle": True, "mmap": False}

def try_convert_object_array(arr_obj):
    """
    支持两种常见结构：
    - shape (N,) 且每个元素是 (d,) 向量
    - shape (d, N) 但 dtype=object（每个单元是标量或0d数组）
    返回 (matrix(d, N), info_str)
    """
    if arr_obj.dtype != object:
        return None, "not object"

    shape = arr_obj.shape
    # 情况A：一维列表，每个元素一个 (d,) 向量
    if len(shape) == 1 and shape[0] > 0 and hasattr(arr_obj[0], "shape"):
        first = arr_obj[0]
        if len(first.shape) == 1:
            d = first.shape[0]
            N = len(arr_obj)
            # 堆叠为 (N, d)，然后转置成 (d, N)
            mat = np.stack(arr_obj, axis=0).T.astype(np.float32, copy=False)
            return mat, f"stacked from list of {N} vectors of dim {d}"
    # 情况B：看上去像 (d, N) 但 dtype=object
    if len(shape) == 2:
        d, N = shape
        # 尝试把每个元素转成 float32 再整体 view
        try:
            mat = np.empty(shape, dtype=np.float32)
            # 向量化转换（对大文件可能慢，但一次性修复用）
            for i in range(d):
                for j in range(N):
                    mat[i, j] = float(arr_obj[i, j])
            return mat, f"casted elementwise from object array of shape {shape}"
        except Exception as e:
            return None, f"failed elementwise cast: {e}"

    return None, f"unrecognized object layout: shape={shape}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="把 object K0 转存为 (d,N) float32 矩阵并覆盖原文件（谨慎）")
    args = parser.parse_args()

    for layer, path in LAYER_PATHS.items():
        print(f"\n=== Layer {layer} ===")
        if not os.path.exists(path):
            print(f"  [X] not found: {path}")
            continue
        arr, meta = safe_load(path)
        print(f"  loaded: dtype={arr.dtype}, shape={arr.shape}, meta={meta}")

        if arr.dtype == object:
            print("  -> object array detected; inspecting...")
            conv, why = try_convert_object_array(arr)
            print(f"  convert_probe: {why}")
            if conv is not None:
                d, N = conv.shape
                print(f"  [OK] can convert to numeric (d={d}, N={N}), dtype={conv.dtype}")
                print(f"    sample stats: min={conv.min():.4g} max={conv.max():.4g} "
                      f"||K||_F={np.linalg.norm(conv):.4g}")
                if args.fix:
                    tmp = path + ".fixed.tmp.npy"
                    np.save(tmp, conv.astype(np.float32, copy=False))
                    os.replace(tmp, path)
                    print(f"  [SAVED] rewrote {path} as float32 matrix (d,N)=({d},{N})")
            else:
                print("  [!] cannot auto-convert; 需要看导出逻辑如何保存的。")
        else:
            dN = "x".join(map(str, arr.shape))
            # 数值矩阵的话快速检查
            try:
                finite = np.isfinite(arr[: min(4, arr.shape[0]), : min(4, arr.shape[1])]).all() if arr.ndim==2 else np.isfinite(arr).all()
            except Exception:
                finite = "n/a"
            print(f"  numeric ok; shape={dN}, finite_check={finite}")

if __name__ == "__main__":
    main()
