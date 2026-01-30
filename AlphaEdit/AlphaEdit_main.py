import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

# ---- NEW: global W0 cache for fair comparison across calls ----
GLOBAL_W0 = {}  # key: (model_key, weight_name) -> CPU tensor

# ---- NEW: history KV cache (per layer) for exact term1_hist(W_t) ----
# key: (model_key, weight_name) -> {"K": CPU [d_k, n_tot], "V": CPU [n_tot, d_v]}
HIST_KV = {}


def _mom2_params_from_hparams(hparams: AlphaEditHyperParams):
    mom2_dataset = getattr(hparams, "mom2_dataset", "wikipedia")
    mom2_n_samples = int(getattr(hparams, "mom2_n_samples", 100000))
    mom2_dtype = getattr(hparams, "mom2_dtype", "float32")
    return mom2_dataset, mom2_n_samples, mom2_dtype


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


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm.
    Invariant: model at beginning of function == model at end of function
    """

    # ---- optional: reset history KV cache for a fresh run ----
    if os.getenv("RESET_HIST_KV", "0").strip() == "1":
        HIST_KV.clear()

    # ---- CSV logging config (minimal, no call-site change) ----
    OBJ_CSV_PATH = os.getenv("ALPHAEDIT_OBJ_CSV", "alphaedit_obj_terms.csv")
    csv_exists = os.path.exists(OBJ_CSV_PATH)

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
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

    # Compute z for final layer
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
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                print(f"Cached k/v pair at {cache_fname}")

    zs = torch.stack(z_list, dim=1)  # [d_v, u_t]

    # ---- open CSV once per call, append rows per layer ----
    with open(OBJ_CSV_PATH, "a", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "case_id_first",
                "num_requests",
                "layer_idx",
                "layer",
                # HIST term1 (evaluated at current W on all past samples)
                "term1_hist_sum_KtT_W_minus_Vt_sq",
                # STEP term1 (only current step, for debug)
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

            # Get current model activations
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = zs - cur_zs
            z_err_mean = torch.linalg.norm(targets, dim=0).mean().item()
            print("z error", z_err_mean)

            # Align columns with layer_ks
            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets_rep = targets.repeat_interleave(repeat_factor, dim=1)  # [d_out, n]

            # Algorithm uses distributed residual (keep as-is)
            resid = targets_rep / (len(hparams.layers) - i)  # [d_out, n]

            upd_matrix = torch.linalg.solve(
                P[i, :, :].cuda() @ (layer_ks @ layer_ks.T + cache_c[i, :, :].cuda())
                + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
                P[i, :, :].cuda() @ layer_ks @ resid.T,
            )

            # Adjust update matrix shape
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            orig_norm = torch.linalg.norm(weights[weight_name]).item()
            upd_norm = torch.linalg.norm(upd_matrix).item()
            print("orig norm", orig_norm)
            print("upd norm", upd_norm)

            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix

            # ---- FAIR objective terms ----
            with torch.no_grad():
                W_raw = weights[weight_name]
                W0_raw = GLOBAL_W0[(model_key, weight_name)].to(W_raw.device, W_raw.dtype)

                # K: [d_in, n], V_hist_eval: [n, d_out]  (NOTE: use targets_rep, NOT resid)
                K = layer_ks
                V_hist_eval = targets_rep.T  # [n, d_out]

                # Ensure W_eff is [d_out, d_in]
                if W_raw.shape[1] == K.shape[0]:
                    W_eff = W_raw
                    W0_eff = W0_raw
                elif W_raw.shape[0] == K.shape[0]:
                    W_eff = W_raw.T
                    W0_eff = W0_raw.T
                else:
                    raise RuntimeError(
                        f"Weight shape {tuple(W_raw.shape)} incompatible with K dim {K.shape[0]}"
                    )

                # unify with RLS representation:
                # Wk: [d_k, d_v] = [d_in, d_out]
                Wk = W_eff.T.contiguous()
                W0k = W0_eff.T.contiguous()
                Kk = K.contiguous()                 # [d_in, n]
                Vk_step = V_hist_eval.contiguous()  # [n, d_out]

                # ---- append current step to history cache (CPU) ----
                key_hist = (model_key, weight_name)
                _hist_kv_append(
                    key_hist,
                    Kk.detach().to("cpu", dtype=torch.float32),
                    Vk_step.detach().to("cpu", dtype=torch.float32),
                )

                # ---- STEP term1 (debug): ||K^T W - V||^2 on current step ----
                pred_step = Kk.T.float() @ Wk.float()               # [n, d_out]
                term1_step = ((pred_step - Vk_step.float()) ** 2).sum().item()

                # ---- HIST term1: sum_{all past} ||K^T W_current - V||^2 ----
                chunk_cols = int(os.getenv("OBJ_HIST_CHUNK", "512"))
                term1_hist = _term1_hist_eval(
                    key_hist,
                    W=Wk,
                    device=Wk.device,
                    chunk_cols=chunk_cols,
                )

                # term2: || W - W0_global ||^2
                delta = (Wk - W0k).float()
                term2 = (delta ** 2).sum().item()

                # term3: trace(delta^T G0 delta), G0 from mom2 stats (UNWEIGHTED)
                mom2_dataset, mom2_n_samples, mom2_dtype = _mom2_params_from_hparams(hparams)
                layer_name_for_stats = hparams.rewrite_module_tmp.format(layer)
                G0 = get_cov(
                    model=model,
                    tok=tok,
                    layer_name=layer_name_for_stats,
                    mom2_dataset=mom2_dataset,
                    mom2_n_samples=mom2_n_samples,
                    mom2_dtype=mom2_dtype,
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

                    "term1_hist_sum_KtT_W_minus_Vt_sq": term1_hist,
                    "term1_step_KtT_W_minus_Vt_sq": term1_step,

                    "term2_W_minus_W0global_sq": term2,
                    "term3_trace_deltaT_G0_delta": term3,

                    "orig_weight_norm": orig_norm,
                    "upd_norm": upd_norm,
                    "z_error_mean": z_err_mean,
                }
            )
            f_csv.flush()

            # Clear GPU memory
            for x in [layer_ks, cur_zs, targets, targets_rep, upd_matrix]:
                try:
                    x.cpu()
                except Exception:
                    pass
                del x
            torch.cuda.empty_cache()

    # keep your original cache_c update (algorithm-related; NOT used for term3 anymore)
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


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
    Retrieves mom2 moment statistics (approx K0^T K0), caches result.
    Returns CUDA tensor; inv=True returns its inverse.
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

    return torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")


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
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
