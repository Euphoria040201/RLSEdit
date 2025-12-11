import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from memit.memit_seq_main import apply_memit_seq_to_model
from memit.memit_rect_main import apply_memit_rect_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from EvoEdit import EvoEditHyperParams
from EvoEdit.EvoEdit_main import apply_EvoEdit_to_model, get_cov
from RLSEdit import RLSEditHyperParams
from RLSEdit.RLSEdit_main import apply_RLSEdit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from nse import NSEHyperParams
from nse.nse_main import apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "EvoEdit": (EvoEditHyperParams, apply_EvoEdit_to_model),
    "RLSEdit": (RLSEditHyperParams, apply_RLSEdit_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "NSE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}



def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    forgetting_eval_interval: int = 0,
    save_every: int = 0,                 
    checkpoint_subdir: str = "checkpoints", 
    downstream_eval_steps: int = 0
):

    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    ckpt_root = run_dir / checkpoint_subdir
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if "MEMIT" in alg_name: 
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit", "EvoEdit", "RLSEdit","MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "NSE":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
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
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["EvoEdit","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            if alg_name == "AlphaEdit" or alg_name == "EvoEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        else:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit" or alg_name == "EvoEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    if alg_name == "AlphaEdit" or alg_name == "EvoEdit":
        for i, layer in enumerate(hparams.layers):
            proj = get_project(model,tok,layer,hparams)
            P[i,:,:] = proj
        torch.save(P, "null_space_project.pt")


    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    
    ## [ADDED] List to store records for the forgetting evaluation
    history_of_edits = []

    cnt = 0
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"=================================================================={cnt+1}_edit==================================================================")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            ## [ADDED] Also store the records in history even if skipping computation
            history_of_edits.extend(record_chunks)
            cnt += 1
            continue
        
        ## [ADDED] Store the records for this chunk to test against later.
        history_of_edits.extend(record_chunks)

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT", "EvoEdit", "RLSEdit","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "EvoEdit", "MEMIT_seq", "NSE"]) else dict()
        if alg_name == "AlphaEdit":
            nc_args = dict(P=P)
        elif alg_name == "EvoEdit":
            nc_args = dict(P=P)
        else:
            nc_args = dict()
        if cnt == 0 and downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
            glue_results = {'edit_num': -1}

            out_file = glue_save_location + "base.json"
            
            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE", "EvoEdit", "RLSEdit"]):
            result = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    **args_conserve_memory,
                    **etc_args,
                    **seq_args,
                    **nc_args,
                )
            if alg_name == "AlphaEdit":
                edited_model, cache_c = result
            else:
                edited_model, cache_c = result
            if "cache_c" in seq_args:
                seq_args["cache_c"] = cache_c
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=False,
                    **args_conserve_memory,
                    **etc_args,
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start
        model = edited_model
        cnt+=1
        print("Execution took", exec_time)
        if save_every != 0:
            total_edits = cnt * num_edits
            if save_every > 0 and total_edits % save_every == 0:
                ckpt_dir = ckpt_root / f"edits_{total_edits:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                print(f"[Checkpoint] Saving model at {ckpt_dir} after {total_edits} edits ...")
                try:
                    model.save_pretrained(
                        ckpt_dir,
                        safe_serialization=True,
                        max_shard_size="5GB"
                    )
                    tok.save_pretrained(ckpt_dir)
                    with open(ckpt_dir / "meta.json", "w") as f:
                        json.dump({
                            "total_edits": int(total_edits),
                            "last_case_ids": case_ids,
                            "layers": hparams.layers,
                            "rewrite_module_tmp": hparams.rewrite_module_tmp,
                            "exec_time_sec_last_chunk": float(exec_time)
                        }, f, indent=2)
                except Exception as e:
                    print(f"[Checkpoint][WARN] save failed: {e}")
        # ----------------------------------------------------

                ## --- [ADDED] FORGETTING EVALUATION BLOCK (REVISED) ---
        total_edits = cnt * num_edits
        if forgetting_eval_interval > 0 and total_edits > 0 and total_edits % forgetting_eval_interval == 0:
            print(f"--- Running Forgetting Evaluation at {total_edits} Edits ---")

            forgetting_results = [] # will hold per-case dicts with three accuracies

            # 用当前模型评测所有历史 edits
            for i, past_record in enumerate(history_of_edits):
                # 若想彻底跳过慢生成，可把 snips, vec 改成 None, None
                metrics = ds_eval_method(
                    edited_model,
                    tok,
                    past_record,
                    snips,
                    vec
                )

                # 仅调试用：看前 3 个样本的 metrics 结构，帮助确认提取路径
                if i < 3:
                    try:
                        print("[DEBUG] sample metrics:", json.dumps(metrics, indent=2)[:1200])
                    except Exception:
                        print("[DEBUG] sample metrics (non-json-serializable keys):", type(metrics), list(metrics.keys()) if isinstance(metrics, dict) else "not a dict")

                # 1) rewrite accuracy
                rewrite_corr = np.mean(metrics["rewrite_prompts_correct"])
                # 2) paraphrase accuracy
                para_corr    = np.mean(metrics["paraphrase_prompts_correct"])
                # 3) neighborhood accuracy
                neigh_corr   = np.mean(metrics["neighborhood_prompts_correct"])


                forgetting_results.append({
                    "case_id": past_record.get("case_id", -1),
                    "rewrite_acc": float(rewrite_corr),
                    "paraphrase_acc": float(para_corr),
                    "neighborhood_acc": float(neigh_corr)
                })

            if forgetting_results:
                avg_rewrite  = float(np.mean([r["rewrite_acc"]     for r in forgetting_results]))
                avg_para     = float(np.mean([r["paraphrase_acc"]  for r in forgetting_results]))
                avg_neigh    = float(np.mean([r["neighborhood_acc"] for r in forgetting_results]))
                # 保存报告
                report_path = run_dir / f"forgetting_report_at_{total_edits}_edits.json"
                summary_report = {
                    "checkpoint_total_edits": int(total_edits),
                    "num_edits_tested": int(len(history_of_edits)),
                    "average_rewrite_accuracy":  f"{avg_rewrite:.4f}",
                    "average_paraphrase_accuracy": f"{avg_para:.4f}",
                    "average_neighborhood_accuracy": f"{avg_neigh:.4f}",
                    "detailed_results": forgetting_results
                }
                with open(report_path, "w") as f:
                    json.dump(summary_report, f, indent=2)

                print(f"→ rewrite avg: {avg_rewrite:.4f}, paraphrase avg: {avg_para:.4f}, neighborhood avg: {avg_neigh:.4f}")
                print(f"Forgetting report saved to {report_path}")
            print("---------------------------------------------------------")
        ## --- END OF ADDED BLOCK ---

        '''
        # Evaluate new model
        if downstream_eval_steps > 0 and cnt % args.downstream_eval_steps == 0:
            glue_results = {
                            'edit_num': cnt*num_edits,
                            'case_id': case_ids
                            }

            out_file = glue_save_location + "case_{}.json".format(record["case_id"])#stores the last case ID of the batch

            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                        
            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        '''

    start = time()
    gen_test_vars = [snips, vec]
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            ),
        }
        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

    print("Evaluation took", time() - start)

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    with torch.no_grad():
        s = S.detach().cpu().numpy()
        cnt = int((s < threshold).sum())
        print(f"[nullspace] layer={layer} dim={s.size}, "
            f"min={s.min():.3e}, median={np.median(s):.3e}, max={s.max():.3e}, "
            f"count(<{threshold:g})={cnt}/{s.size}")
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    proj = U[:, small_singular_indices] @ U[:, small_singular_indices].T
    return proj

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit", "EvoEdit", "MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT", "MEND","NSE","RLSEdit"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    ## [ADDED] New argument for the forgetting evaluation experiment
    parser.add_argument(
        "--forgetting_eval_interval",
        type=int,
        default=0,
        help="Interval (in number of edits) to test accuracy on all previous edits. If 0, this test is disabled. E.g., 100.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=2000,
        help="Save the whole model every N edits (sample-level). 0 = disable."
    )
    parser.add_argument(
        "--checkpoint_subdir",
        type=str,
        default="checkpoints",
        help="Subdirectory under the run dir to store checkpoints."
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        forgetting_eval_interval=args.forgetting_eval_interval, ## [MODIFIED] Pass new argument to main
        save_every=args.save_every,
        checkpoint_subdir=args.checkpoint_subdir,
        downstream_eval_steps=args.downstream_eval_steps,
    )
