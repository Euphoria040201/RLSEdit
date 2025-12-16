from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any  # ← 确保有 Optional/Dict/Any


from util.hparams import HyperParams


@dataclass
class RLSEditHyperParams(HyperParams):
    # Method
    model_name: str
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    nullspace_threshold: float
    L2: float
    # ---------- RLS/Woodbury 需要的新增参数 ----------
    # 论文里的 λ、μ，用于 C0 = (λ^2 I + μ^2 K0 K0^T)^{-1}
    lambda_reg: float = 0.0
    mu_reg: float = 0.0
    weight_reg: float = 1.0
    # 可选：每层的 K0（若不给则仅用 λ^2 I 初始化）
    # 形状: K0_by_layer[layer] ∈ R^{d_k × m0}
    K0_path_by_layer: Optional[Dict[int, str]] = None