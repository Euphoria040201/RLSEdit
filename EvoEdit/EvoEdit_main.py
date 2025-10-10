import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.linalg import svd, qr
from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .EvoEdit_hparams import EvoEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

update_timing = {
    "per_layer": [],    
    "total_solve_s": 0.0,
    "total_proj_s": 0.0,     
    "total_all_s": 0.0        
}
def pk_norm_small(P_i, K_t, eps=1e-4):
    with torch.no_grad():
        val = torch.linalg.norm(P_i @ K_t, ord='fro')
        #print("P_i @ K_t = ", val)
    return bool(val.item() < eps)

def update_projector_from_P_and_K(P0: torch.Tensor, Knew: torch.Tensor, tol: float = 1e-10):
    device, dtype = P0.device, P0.dtype
    R = P0 @ Knew.to(device=device, dtype=dtype)

    if R.numel() == 0 or R.shape[1] == 0:
        return P0, torch.zeros((P0.size(0), 0), device=device, dtype=dtype)

    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    eps  = torch.finfo(S.dtype).eps
    thr  = max(R.shape) * eps * (S[0] if S.numel() else torch.tensor(0., device=device, dtype=dtype))
    r    = int((S > torch.maximum(thr, torch.tensor(tol, device=device, dtype=dtype))).sum().item())

    if r == 0:
        return P0, torch.zeros((P0.size(0), 0), device=device, dtype=dtype)

    Q_add = U[:, :r]
    P_new = P0 - Q_add @ Q_add.transpose(-2, -1)
    return P_new, Q_add

def apply_EvoEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EvoEditHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
    apply_woodbury = True,
    z_error_threshold = 1e-2,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    print("Using EvoEdit.")
    global update_timing
    t_all0 = time.perf_counter()

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
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
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
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
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
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
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)
    skipped_this_request = False
    total_s_all_layers = 0
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

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
        z_error = torch.linalg.norm(targets, dim=0).mean()
        print("z error", z_error)
        
        
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        
        
        resid = targets / (len(hparams.layers) - i)  # d×r
        if apply_woodbury:
            if z_error < z_error_threshold:
                print("Skip due to small z_error")
                skipped_this_request=True
                break
            print("Applying Woodbury Method")
            # ----- Low-rank (Woodbury) with L2 -----
            K_t = layer_ks.to("cuda")
            P_i = P[i, :, :].to("cuda")
            if pk_norm_small(P_i, K_t, eps=1e-3):
                print(f"[Skip] ||P K||_F is tiny at layer {layer}, skip edit for this request")
                skipped_this_request=True
                break
            t_solve0 = time.perf_counter()

            Y   = P_i @ K_t                      # d×r
            S0  = K_t.T @ Y                      # r×r = K^T P K
            r   = S0.size(0)
            Ir  = torch.eye(r, device=S0.device, dtype=S0.dtype)
            eps = 1e-6 if S0.dtype == torch.float32 else 1e-12
            try:
                # Cholesky on (lam I + S0)
                L = torch.linalg.cholesky(S0 + hparams.L2 * Ir + eps * Ir)   # lower, r×r

                # Solve (lam I + S0) B = Y^T  ->  B = (lam I + S0)^{-1} Y^T
                YT = Y.T                                              # r×d
                B  = torch.cholesky_solve(YT, L)                      # r×d

                # Δ = resid @ B
                upd_matrix = resid @ B
            except RuntimeError as e:
                print(e) 
                skipped_this_request=True
                print("Failed for cholesky. Skip this edit.")
                break
        else:
            d = layer_ks.shape[0]
            Id = torch.eye(d, device="cuda", dtype=layer_ks.dtype)
            upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T) + hparams.L2 * Id,
                P[i,:,:].cuda() @ layer_ks @ resid.T,
            )
            

        solve_s = time.perf_counter() - t_solve0
        update_timing["total_solve_s"] += solve_s
        t_proj0 = time.perf_counter()
        #update projector
        P[i,:,:], _ = update_projector_from_P_and_K(P[i,:,:], layer_ks.detach())

        proj_s = time.perf_counter() - t_proj0
        update_timing["total_proj_s"] += proj_s

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        # Clear GPU memory
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

        layer_total_s = proj_s + solve_s
        total_s_all_layers += layer_total_s
        update_timing["per_layer"].append({
            "layer": int(layer),
            "solve_s": solve_s,
            "proj_update_s": proj_s,
            "layer_total_s": layer_total_s
        })
        print(f"[Timing] layer {layer}: solve={solve_s:.4f}s, proj_update={proj_s:.4f}s, total={layer_total_s:.4f}s")


    if not skipped_this_request:
        for i, layer in enumerate(hparams.layers):
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    update_timing["total_all_s"] += total_s_all_layers
    print(f"[Timing][Totals] solve={update_timing['total_solve_s']:.4f}s, "
          f"proj_update={update_timing['total_proj_s']:.4f}s, "
          f"all={update_timing['total_all_s']:.4f}s")

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
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
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