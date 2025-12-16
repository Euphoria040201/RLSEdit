import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
import time
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import svd, qr
from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .RLSEdit_hparams import RLSEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

# ---- 简单计时容器（新增）----
update_timing = {
    "per_layer": [],            # 每层明细
    "total_train_z_s": 0.0,     # 训练/生成 zs 的总时长（compute_z）
    "total_cur_z_s": 0.0,       # 计算 cur_z（forward 抽取）的总时长
    "total_update_s": 0.0,      # 求解/更新权重的总时长
    "total_all_s": 0.0,         # 三者之和
}


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _robust_spd_inverse(
    matrix: torch.Tensor,
    layer_idx: int,
    base_eps: float = 1e-6,
    max_tries: int = 5,
) -> torch.Tensor:
    """
    尝试 Cholesky，若失败则逐步添加对角抖动；仍失败则回退到 pinv。
    这样避免奇异/非正定导致的崩溃。
    """
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    mat_sym = _symmetrize(matrix)
    # 用平均对角线做缩放，防止全零矩阵时抖动过小
    diag_scale = mat_sym.diagonal().abs().mean().clamp(min=1.0).item()
    jitter = base_eps * diag_scale

    for attempt in range(max_tries):
        try:
            R = torch.linalg.cholesky(mat_sym)
            return torch.cholesky_inverse(R, upper=False)
        except Exception as e:
            print(
                f"[SOLVE][layer {layer_idx}] cholesky failed ({e}); "
                f"adding jitter {jitter:.3e} (attempt {attempt + 1}/{max_tries})"
            )
            mat_sym = mat_sym + eye * jitter
            jitter *= 10.0

    print(
        f"[SOLVE][layer {layer_idx}] cholesky still failing after retries; "
        "using pseudo-inverse as fallback"
    )
    return torch.linalg.pinv(mat_sym)


def apply_RLSEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: RLSEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,        # 存放的是 C_t = (A_t^T A_t)^{-1}
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    严格 RLS/Woodbury 实现（按 4.4）：
      C0 = (lambda^2 I + mu^2 K0^T K0)^(-1)
      F_t = C_{t-1} K_t^T
      C_t = C_{t-1} - F_t (I + K_t F_t)^(-1) F_t^T
      W_t* = W_{t-1}* + C_t K_t^T (V_t - K_t W_{t-1}*)
    """
    global update_timing
    update_timing = {
        "per_layer": [],
        "total_train_z_s": 0.0,
        "total_cur_z_s": 0.0,
        "total_update_s": 0.0,
        "total_all_s": 0.0,
    }
    t_all0 = time.perf_counter()
    print("Applying RLS Edit.")
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []
    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")
        if not data_loaded:
            t_train_z0 = time.perf_counter()
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            train_z_s = time.perf_counter() - t_train_z0
            update_timing["total_train_z_s"] += train_z_s
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)  # [d_v, u_t]；这里令 u_t = n_req（与 K_t 的列数一致）

    # ---- 初始化每层的 C_0 与 W*_0 （严格）----
    lam_env = os.environ.get("RLSEDIT_LAMBDA_REG", "").strip()
    lam = (
        float(lam_env)
        if lam_env
        else float(getattr(hparams, "lambda_reg", 0.0))
    )
    mu  = float(getattr(hparams, "mu_reg", 0.0))
    Wstar_state = {}
    mom2_update_weight = float(
        os.environ.get("RLSEDIT_MOM2_UPDATE_WEIGHT", hparams.mom2_update_weight)
    )
    print(f"[RLS] mom2_update_weight={mom2_update_weight}")
    print(f"[RLS] lambda_reg={lam}")

    if cache_c is None:
        C_list = []
        for lyr in hparams.layers:
            # 探测 d_k
            _probe = compute_ks(model, tok, requests, hparams, lyr, context_templates).T
            d_k = _probe.shape[0]
            del _probe

            I = torch.eye(d_k, dtype=torch.float32, device="cuda")

            # 从导出 K0 构造严格的 K0^T K0  -> [d_k, d_k]
            layer_name_for_stats = hparams.rewrite_module_tmp.format(lyr)
            Cov = get_cov(
                model=model,
                tok=tok,
                layer_name=layer_name_for_stats,
                mom2_dataset=hparams.mom2_dataset,
                mom2_n_samples=hparams.mom2_n_samples,
                mom2_dtype=hparams.mom2_dtype,
                inv=False,
                force_recompute=False,
            ).to(device="cuda", dtype=torch.float32) 
            Cov = Cov * mom2_update_weight
            M0 = (lam**2) * I + (mu**2) * Cov  # 先按公式构造，求逆阶段再按需抖动

            # C0 = M0^{-1}（稳健求解：自适应抖动 + pinv 兜底）
            C0 = _robust_spd_inverse(M0, layer_idx=lyr)

            C_list.append(C0)

        C = torch.stack(C_list, dim=0)  # [n_layers, d_k, d_k]（CPU 缓存）
    else:
        C = cache_c.clone()

    # ----------------------- 主循环（严格递推） -----------------------
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"

        # K_t: [d_k, u_t]
        K_t = compute_ks(model, tok, requests, hparams, layer, context_templates).T.to("cuda")
        print(f"Writing {K_t.size(1)} key/value pair(s) into layer {layer}")
        Kt_norm = K_t.norm().item()
        Kt_max = K_t.abs().max().item() if K_t.numel() > 0 else 0.0
        print(f"[DEBUG][layer {layer}] ‖K_t‖2={Kt_norm:.6f} max|K_t|={Kt_max:.6f}")
        if not torch.isfinite(K_t).all() or Kt_norm == 0.0:
            raise RuntimeError(f"K_t invalid at layer {layer}: norm={Kt_norm}, finite={bool(torch.isfinite(K_t).all())}")

        # V_t: [u_t, d_v]（严格：不做层间平均、不做 repeat）
        t_curz0 = time.perf_counter()
        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T.to("cuda")  # [d_v, u_t]
        cur_z_s = time.perf_counter() - t_curz0
        update_timing["total_cur_z_s"] += cur_z_s
        targets = (zs.to("cuda") - cur_zs)           # [d_v, u_t]
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        V_t = targets.T                               # [u_t, d_v]
        repeat_factor = (K_t.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        # W*_0 初始化（与权重形状一致）
        if (layer not in Wstar_state) or (Wstar_state[layer] is None):
            W_cur = weights[weight_name].detach().to("cpu")
            d_k_, d_v_ = K_t.size(0), V_t.size(1)
            if W_cur.shape == (d_k_, d_v_):
                W_prev0 = W_cur
            elif W_cur.T.shape == (d_k_, d_v_):
                W_prev0 = W_cur.T
            else:
                W_prev0 = torch.zeros((d_k_, d_v_), dtype=W_cur.dtype)
            Wstar_state[layer] = W_prev0

        C_prev = C[i].to(device=K_t.device, dtype=K_t.dtype)
        W_prev = Wstar_state[layer].to(K_t.device)

        # Woodbury（严格）：F_t, S_t, C_t
        t_update0 = time.perf_counter()
        F_t = C_prev @ K_t                              # [d_k, u_t]
        S_t = torch.eye(K_t.size(1), dtype=K_t.dtype, device=K_t.device) + K_t.T @ F_t  # [u_t, u_t]
        R_small = torch.linalg.cholesky(S_t, upper=True)
        G_t = torch.linalg.solve_triangular(R_small, F_t, upper=True, left=False)
        C_t = C_prev - G_t @ G_t.T                      # [d_k, d_k]
        C[i] = C_t.detach().to("cpu")

        # W*_t 递推（严格）
        resid_y = V_t - K_t.T @ W_prev                  # [u_t, d_v]
        resid_y = resid_y / (len(hparams.layers) - i)
        W_t = W_prev + C_t @ K_t @ resid_y              # [d_k, d_v]

        delta_W = W_t - W_prev
        print(f"[delta_W] shape={tuple(delta_W.shape)} dtype={delta_W.dtype} device={delta_W.device}")
        print(f"[delta_W] frob_norm={torch.linalg.norm(delta_W).item():.6e} "
              f"max_abs={delta_W.abs().max().item():.6e}")

        upd_matrix = upd_matrix_match_shape(delta_W, weights[weight_name].shape)
        upd_matrix = upd_matrix.to(weights[weight_name].device, weights[weight_name].dtype)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        Wstar_state[layer] = W_t.detach().to("cpu")
        update_s = time.perf_counter() - t_update0
        update_timing["total_update_s"] += update_s

        # 清理
        for x in [K_t, cur_zs, targets, V_t, F_t, S_t, R_small, G_t, delta_W, W_t]:
            x = None
        torch.cuda.empty_cache()

        layer_wall_s = cur_z_s + update_s
        update_timing["per_layer"].append(
            {
                "layer": int(layer),
                "cur_z_s": cur_z_s,
                "update_s": update_s,
                "layer_total_s": layer_wall_s,
            }
        )
        print(
            f"[Timing] layer {layer}: cur_z={cur_z_s:.4f}s, "
            f"update={update_s:.4f}s, total={layer_wall_s:.4f}s"
        )

    update_timing["total_all_s"] = (
        update_timing["total_train_z_s"]
        + update_timing["total_cur_z_s"]
        + update_timing["total_update_s"]
    )
    wall_elapsed = time.perf_counter() - t_all0
    print(
        f"[Timing][Totals] train_z={update_timing['total_train_z_s']:.4f}s, "
        f"cur_z={update_timing['total_cur_z_s']:.4f}s, "
        f"update={update_timing['total_update_s']:.4f}s, "
        f"all={update_timing['total_all_s']:.4f}s, "
        f"wall={wall_elapsed:.4f}s"
    )
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return model, C



def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    与 MEMIT 相同：用 layer_stats(mom2) 提取协方差（第二矩，未归一化的 E[kk^T]），做缓存。
    返回：CUDA 张量；若 inv=True 则返回其逆。
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        # 与 MEMIT 一致：直接拿 moment() 作为 covariance 矩阵（形状 [d_k, d_k]）
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )
