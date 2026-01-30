import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
import time
import pdb
import resource
import sys
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
GLOBAL_W0 = {}  # key: (model_key, weight_name) -> CPU tensor

# ---- NEW: history KV cache (per layer) for exact term1_hist(W_t) ----
# key: (model_key, weight_name) -> {"K": CPU [d_k, n_tot], "V": CPU [n_tot, d_v]}
HIST_KV = {}

# ---- 简单计时容器（新增）----
update_timing = {
    "per_layer": [],
    "total_train_z_s": 0.0,
    "total_cur_z_s": 0.0,
    "total_update_s": 0.0,
    "total_all_s": 0.0,
}


def _format_bytes(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _cpu_max_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def _log_memory_stats(tag: str = "MEM") -> None:
    cpu_rss = _cpu_max_rss_bytes()
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        print(
            f"[{tag}] CPU_RSS_MAX={_format_bytes(cpu_rss)} "
            f"GPU_ALLOC={_format_bytes(allocated)} "
            f"GPU_RESERVED={_format_bytes(reserved)} "
            f"GPU_MAX_ALLOC={_format_bytes(max_allocated)} "
            f"GPU_MAX_RESERVED={_format_bytes(max_reserved)}"
        )
    else:
        print(f"[{tag}] CPU_RSS_MAX={_format_bytes(cpu_rss)} GPU=unavailable")


def _hist_kv_append(key, K_cpu: torch.Tensor, V_cpu: torch.Tensor):
    """
    K_cpu: [d_k, n] on CPU
    V_cpu: [n, d_v] on CPU
    """
    K_cpu = K_cpu.contiguous()
    V_cpu = V_cpu.contiguous()
    if key not in HIST_KV:
        HIST_KV[key] = {"K": K_cpu, "V": V_cpu}
    else:
        HIST_KV[key]["K"] = torch.cat([HIST_KV[key]["K"], K_cpu], dim=1)
        HIST_KV[key]["V"] = torch.cat([HIST_KV[key]["V"], V_cpu], dim=0)


@torch.no_grad()
def _term1_hist_eval(
    key,
    W: torch.Tensor,
    device: torch.device,
    chunk_cols: int = 512,
) -> float:
    """
    Exact: sum_{hist} ||K^T W - V||^2 evaluated at CURRENT W.
    W: [d_k, d_v] on GPU/device
    """
    K_hist = HIST_KV[key]["K"]  # CPU [d_k, n_tot]
    V_hist = HIST_KV[key]["V"]  # CPU [n_tot, d_v]
    n_tot = K_hist.shape[1]

    acc = 0.0
    Wf = W.to(device=device, dtype=torch.float32)

    for s in range(0, n_tot, chunk_cols):
        e = min(s + chunk_cols, n_tot)
        Kc = K_hist[:, s:e].to(device=device, dtype=torch.float32, non_blocking=True)  # [d_k, c]
        Vc = V_hist[s:e].to(device=device, dtype=torch.float32, non_blocking=True)    # [c, d_v]
        pred = Kc.T @ Wf                                                               # [c, d_v]
        acc += ((pred - Vc) ** 2).sum().item()

    return float(acc)


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
    """
    eye = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    mat_sym = _symmetrize(matrix)
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
    cache_c=None,        # 存放的是 C_t = (A_t^T A_t)^{-1}
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    严格 RLS/Woodbury 实现（按 4.4）：
      C0 = (lambda^2 I + mu^2 K0^T K0)^(-1)
      F_t = C_{t-1} K_t
      C_t = C_{t-1} - F_t (I + K_t^T F_t)^(-1) F_t^T
      W_t* = W_{t-1}* + C_t K_t (V_t - K_t^T W_{t-1}*)
    """
    global update_timing, GLOBAL_W0, HIST_KV

    # optional: reset history kv
    if os.getenv("RESET_HIST_KV", "0").strip() == "1":
        HIST_KV.clear()

    update_timing = {
        "per_layer": [],
        "total_train_z_s": 0.0,
        "total_cur_z_s": 0.0,
        "total_update_s": 0.0,
        "total_all_s": 0.0,
    }
    t_all0 = time.perf_counter()
    print("Applying RLS Edit.")

    # ---- CSV logging config ----
    OBJ_CSV_PATH = os.getenv("RLSEDIT_OBJ_CSV", "rlsedit_obj_terms.csv")
    csv_exists = os.path.exists(OBJ_CSV_PATH)

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

    # ---- NEW: global initial W0 (true W0 before ANY edits) ----
    model_key = model.config._name_or_path.replace("/", "_")
    for name, p in weights.items():
        k = (model_key, name)
        if k not in GLOBAL_W0:
            GLOBAL_W0[k] = p.detach().cpu().clone()

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

    zs = torch.stack(z_list, dim=1)  # [d_v, u_t]

    # ---- 初始化每层的 C_0 与 W*_0 ----
    lam_env = os.environ.get("RLSEDIT_LAMBDA_REG", "").strip()
    lam = float(lam_env) if lam_env else float(getattr(hparams, "lambda_reg", 0.0))
    mu = float(getattr(hparams, "mu_reg", 0.0))

    Wstar_state = {}
    mom2_update_weight = float(
        os.environ.get("RLSEDIT_MOM2_UPDATE_WEIGHT", hparams.mom2_update_weight)
    )
    print(f"[RLS] mom2_update_weight={mom2_update_weight}")
    print(f"[RLS] lambda_reg={lam}")

    if cache_c is None:
        C_list = []
        for lyr in hparams.layers:
            _probe = compute_ks(model, tok, requests, hparams, lyr, context_templates).T
            d_k = _probe.shape[0]
            del _probe

            I = torch.eye(d_k, dtype=torch.float32, device="cuda")

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

            Cov_alg = Cov * mom2_update_weight
            M0 = (lam**2) * I + (mu**2) * Cov_alg
            C0 = _robust_spd_inverse(M0, layer_idx=lyr)
            C_list.append(C0)

        C = torch.stack(C_list, dim=0)
    else:
        C = cache_c.clone()

    # ---- open CSV once per call ----
    with open(OBJ_CSV_PATH, "a", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "case_id_first",
                "num_requests",
                "layer_idx",
                "layer",
                # NOTE: 保持原列名，但写进去的是 term1_hist（历史和）
                "term1_data_KtT_W_minus_Vt_sq",
                # 额外给一个当前 step 的 term1 方便 debug
                "term1_step_KtT_W_minus_Vt_sq",
                "term2_W_minus_W0global_sq",
                "term3_trace_deltaT_G0_delta",
                "orig_weight_norm",
                "upd_norm",
                "z_error_mean",
            ],
        )
        if not csv_exists:
            writer.writeheader()

        case_id_first = requests[0].get("case_id", -1) if len(requests) > 0 else -1
        num_requests = len(requests)

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
                raise RuntimeError(
                    f"K_t invalid at layer {layer}: norm={Kt_norm}, finite={bool(torch.isfinite(K_t).all())}"
                )

            # V_t: [u_t, d_v]
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

            targets = (zs.to("cuda") - cur_zs)  # [d_v, u_t]
            z_err_mean = torch.linalg.norm(targets, dim=0).mean().item()
            print("z error", z_err_mean)

            V_t = targets.T  # [u_t, d_v]

            # W*_0 初始化
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
            W_prev = Wstar_state[layer].to(K_t.device)  # [d_k, d_v]

            # Woodbury
            t_update0 = time.perf_counter()
            F_t = C_prev @ K_t                               # [d_k, u]
            S_t = torch.eye(K_t.size(1), dtype=K_t.dtype, device=K_t.device) + K_t.T @ F_t
            R_small = torch.linalg.cholesky(S_t, upper=True)
            G_t = torch.linalg.solve_triangular(R_small, F_t, upper=True, left=False)
            C_t = C_prev - G_t @ G_t.T
            C[i] = C_t.detach().to("cpu")

            # W*_t
            resid_y = V_t - K_t.T @ W_prev
            resid_y = resid_y / (len(hparams.layers) - i)
            W_t = W_prev + C_t @ K_t @ resid_y  # [d_k, d_v]

            delta_W = W_t - W_prev
            print(f"[delta_W] shape={tuple(delta_W.shape)} dtype={delta_W.dtype} device={delta_W.device}")
            print(f"[delta_W] frob_norm={torch.linalg.norm(delta_W).item():.6e} "
                  f"max_abs={delta_W.abs().max().item():.6e}")

            upd_matrix = upd_matrix_match_shape(delta_W, weights[weight_name].shape)
            upd_matrix = upd_matrix.to(weights[weight_name].device, weights[weight_name].dtype)

            orig_norm = torch.linalg.norm(weights[weight_name]).item()
            upd_norm = torch.linalg.norm(upd_matrix).item()
            print("orig norm", orig_norm)
            print("upd norm", upd_norm)

            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix

            Wstar_state[layer] = W_t.detach().to("cpu")
            update_s = time.perf_counter() - t_update0
            update_timing["total_update_s"] += update_s

            # ---- FAIR objective terms (term1 = HIST sum, evaluated at latest W) ----
            with torch.no_grad():
                W_raw = weights[weight_name]
                W0_raw = GLOBAL_W0[(model_key, weight_name)].to(W_raw.device, W_raw.dtype)

                d_k = K_t.size(0)
                d_v = V_t.size(1)

                # map actual weight to W_eff: [d_k, d_v]
                if W_raw.shape == (d_k, d_v):
                    W_eff = W_raw
                    W0_eff = W0_raw if W0_raw.shape == (d_k, d_v) else W0_raw.T
                elif W_raw.T.shape == (d_k, d_v):
                    W_eff = W_raw.T
                    W0_eff = W0_raw.T if W0_raw.T.shape == (d_k, d_v) else W0_raw
                else:
                    raise RuntimeError(
                        f"[OBJ] weight shape {tuple(W_raw.shape)} incompatible with (d_k,d_v)=({d_k},{d_v})"
                    )

                # ---- append current step to history cache (CPU, un-distributed V) ----
                key_hist = (model_key, weight_name)
                _hist_kv_append(
                    key_hist,
                    K_t.detach().to("cpu", dtype=torch.float32),
                    V_t.detach().to("cpu", dtype=torch.float32),
                )

                # term1_step: only current step
                pred_step = K_t.T.float() @ W_eff.float()  # [u, d_v]
                term1_step = ((pred_step - V_t.float()) ** 2).sum().item()

                # term1_hist: all past steps, evaluated with current W_eff
                chunk_cols = int(os.getenv("OBJ_HIST_CHUNK", "512"))
                term1_hist = _term1_hist_eval(
                    key_hist,
                    W=W_eff,
                    device=W_eff.device,
                    chunk_cols=chunk_cols,
                )

                # term2: ||W - W0_global||^2
                delta = (W_eff - W0_eff).float()
                term2 = (delta ** 2).sum().item()

                # term3: trace(delta^T G0 delta), G0 = mom2 moment (UNWEIGHTED)
                layer_name_for_stats = hparams.rewrite_module_tmp.format(layer)
                G0 = get_cov(
                    model=model,
                    tok=tok,
                    layer_name=layer_name_for_stats,
                    mom2_dataset=hparams.mom2_dataset,
                    mom2_n_samples=hparams.mom2_n_samples,
                    mom2_dtype=hparams.mom2_dtype,
                    inv=False,
                    force_recompute=False,
                ).to(device=delta.device, dtype=torch.float32)

                term3 = ((G0 @ delta) * delta).sum().item()

            print(
                f"[OBJ][layer {layer}] "
                f"term1_step={term1_step:.4e} term1_hist={term1_hist:.4e}  "
                f"term2={term2:.4e}  "
                f"term3={term3:.4e}"
            )

            writer.writerow(
                {
                    "case_id_first": case_id_first,
                    "num_requests": num_requests,
                    "layer_idx": i,
                    "layer": int(layer),
                    # 保持原列名但写 hist
                    "term1_data_KtT_W_minus_Vt_sq": term1_hist,
                    "term1_step_KtT_W_minus_Vt_sq": term1_step,
                    "term2_W_minus_W0global_sq": term2,
                    "term3_trace_deltaT_G0_delta": term3,
                    "orig_weight_norm": orig_norm,
                    "upd_norm": upd_norm,
                    "z_error_mean": z_err_mean,
                }
            )
            f_csv.flush()

            # cleanup
            for x in [K_t, cur_zs, targets, V_t, F_t, S_t, R_small, G_t, delta_W, W_t]:
                x = None
            torch.cuda.empty_cache()

            layer_wall_s = cur_z_s + update_s
            update_timing["per_layer"].append(
                {"layer": int(layer), "cur_z_s": cur_z_s, "update_s": update_s, "layer_total_s": layer_wall_s}
            )
            print(f"[Timing] layer {layer}: cur_z={cur_z_s:.4f}s, update={update_s:.4f}s, total={layer_wall_s:.4f}s")

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
    _log_memory_stats(tag="RLS_MEM")
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
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )
