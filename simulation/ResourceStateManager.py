import numpy as np
from typing import Dict


def init_resource_state(
    serving_ru: np.ndarray,
    prb_total: int,
    ru_prb_cap: int,
    n_ru: int,
    total_tx_power_w: float
) -> Dict:
    """
    Initial RU allocation by UE count:
        RU_PRB_init = (prb_total / total_ue) * ue_count_in_ru

    Input:
        serving_ru: (N,) int
        prb_total: int
        ru_prb_cap: int
        n_ru: int
        total_tx_power_w: float

    Output:
        resource_state: dict
    """
    n_ue = serving_ru.shape[0]
    ue_count_per_ru = np.bincount(serving_ru, minlength=n_ru).astype(np.float64)

    prb_per_ue = prb_total / max(n_ue, 1)
    ru_prb_allocated = prb_per_ue * ue_count_per_ru
    ru_prb_allocated = np.minimum(ru_prb_allocated, float(ru_prb_cap))

    ue_allocated_prb = np.zeros(n_ue, dtype=np.float64)
    ue_power_alloc_w = np.zeros(n_ue, dtype=np.float64)

    for ru_id in range(n_ru):
        mask = serving_ru == ru_id
        n_local = np.sum(mask)
        if n_local > 0:
            ue_allocated_prb[mask] = ru_prb_allocated[ru_id] / n_local
            ue_power_alloc_w[mask] = total_tx_power_w / n_local

    ru_used_prb = compute_ru_used_prb(
        serving_ru=serving_ru,
        ue_allocated_prb=ue_allocated_prb,
        n_ru=n_ru,
    )
    ru_free_prb = ru_prb_allocated - ru_used_prb
    prb_pool_free = float(prb_total) - np.sum(ru_prb_allocated)

    return {
        "ru_prb_allocated": ru_prb_allocated,
        "ru_used_prb": ru_used_prb,
        "ru_free_prb": ru_free_prb,
        "ue_allocated_prb": ue_allocated_prb,
        "ue_power_alloc_w": ue_power_alloc_w,
        "prb_pool_free": prb_pool_free,
        "ue_count_per_ru": ue_count_per_ru,
    }


def compute_ru_used_prb(
    serving_ru: np.ndarray,
    ue_allocated_prb: np.ndarray,
    n_ru: int
) -> np.ndarray:
    """
    Input:
        serving_ru: (N,) int
        ue_allocated_prb: (N,) float
        n_ru: int

    Output:
        ru_used_prb: (M,) float
    """
    ru_used_prb = np.zeros(n_ru, dtype=np.float64)
    for ru_id in range(n_ru):
        mask = serving_ru == ru_id
        ru_used_prb[ru_id] = np.sum(ue_allocated_prb[mask])
    return ru_used_prb


def compute_ru_free_prb(
    ru_prb_allocated: np.ndarray,
    ru_used_prb: np.ndarray
) -> np.ndarray:
    """
    Input:
        ru_prb_allocated: (M,) float
        ru_used_prb: (M,) float

    Output:
        ru_free_prb: (M,) float
    """
    return np.maximum(ru_prb_allocated - ru_used_prb, 0.0).astype(np.float64)


def release_unused_prb(
    ru_prb_allocated: np.ndarray,
    ru_used_prb: np.ndarray,
    prb_pool_free: float
) -> Dict:
    """
    Release all unused RU PRB back to global pool.

    Output:
        ru_prb_allocated_new
        prb_pool_free_new
    """
    ru_free_prb = np.maximum(ru_prb_allocated - ru_used_prb, 0.0)
    released_prb = np.sum(ru_free_prb)

    ru_prb_allocated_new = ru_used_prb.copy()
    prb_pool_free_new = prb_pool_free + released_prb

    return {
        "ru_prb_allocated": ru_prb_allocated_new,
        "prb_pool_free": float(prb_pool_free_new),
        "released_prb": float(released_prb),
    }


def request_prb_for_ru(
    ru_id: int,
    requested_prb: float,
    ru_prb_allocated: np.ndarray,
    prb_pool_free: float,
    ru_prb_cap: float
) -> Dict:
    """
    Input:
        ru_id: int
        requested_prb: float

    Output:
        success: bool
        ru_prb_allocated
        prb_pool_free
        granted_prb
    """
    max_local_add = max(ru_prb_cap - ru_prb_allocated[ru_id], 0.0)
    granted_prb = min(requested_prb, max_local_add, prb_pool_free)

    success = granted_prb >= requested_prb - 1e-9

    ru_prb_allocated_new = ru_prb_allocated.copy()
    ru_prb_allocated_new[ru_id] += granted_prb
    prb_pool_free_new = prb_pool_free - granted_prb

    return {
        "success": bool(success),
        "ru_prb_allocated": ru_prb_allocated_new,
        "prb_pool_free": float(prb_pool_free_new),
        "granted_prb": float(granted_prb),
    }


def estimate_required_prb(
    r_min_bps: np.ndarray,
    serving_gain: np.ndarray,
    ue_power_alloc_w: np.ndarray,
    rb_bandwidth_hz: float
) -> np.ndarray:
    """
    Solve rough PRB demand from:
        R = N_RB * BW_RB * log2(1 + (p / N_RB) * gain)

    This is approximated numerically by scanning candidate RB values.

    Input:
        r_min_bps: (N,) float
        serving_gain: (N,) float
        ue_power_alloc_w: (N,) float
        rb_bandwidth_hz: float

    Output:
        required_prb: (N,) float
    """
    n_ue = r_min_bps.shape[0]
    required_prb = np.ones(n_ue, dtype=np.float64)

    candidate_prb = np.arange(1, 101, dtype=np.float64)

    for i in range(n_ue):
        snr_per_rb = (ue_power_alloc_w[i] / candidate_prb) * serving_gain[i]
        rate = candidate_prb * rb_bandwidth_hz * np.log2(1.0 + snr_per_rb)
        feasible = np.where(rate >= r_min_bps[i])[0]
        if feasible.size > 0:
            required_prb[i] = candidate_prb[feasible[0]]
        else:
            required_prb[i] = candidate_prb[-1]

    return required_prb


def compact_stable_ue_allocation(
    serving_ru: np.ndarray,
    stable_mask: np.ndarray,
    ue_allocated_prb: np.ndarray,
    ue_required_prb: np.ndarray,
    stable_margin_ratio: np.ndarray,
    n_ru: int
) -> Dict:
    """
    Reduce stable UE allocation toward required PRB + margin.

    Input:
        stable_mask: (N,) bool
        ue_allocated_prb: (N,) float
        ue_required_prb: (N,) float
        stable_margin_ratio: (N,) float

    Output:
        ue_allocated_prb_new: (N,) float
        ru_used_prb: (M,) float
    """
    ue_allocated_prb_new = ue_allocated_prb.copy()

    target_prb = ue_required_prb * (1.0 + stable_margin_ratio)
    ue_allocated_prb_new[stable_mask] = np.maximum(target_prb[stable_mask], 1.0)

    ru_used_prb = compute_ru_used_prb(
        serving_ru=serving_ru,
        ue_allocated_prb=ue_allocated_prb_new,
        n_ru=n_ru,
    )

    return {
        "ue_allocated_prb": ue_allocated_prb_new,
        "ru_used_prb": ru_used_prb,
    }


def get_serving_du_cu(
    serving_ru: np.ndarray,
    ru_to_du: np.ndarray,
    du_to_cu: np.ndarray
) -> Dict:
    """
    Input:
        serving_ru: (N,) int
        ru_to_du: (M,) int
        du_to_cu: (D,) int

    Output:
        serving_du: (N,) int
        serving_cu: (N,) int
    """
    serving_du = ru_to_du[serving_ru]
    serving_cu = du_to_cu[serving_du]
    return {
        "serving_du": serving_du.astype(np.int32),
        "serving_cu": serving_cu.astype(np.int32),
    }


def estimate_cpu_requirements(
    r_min_bps: np.ndarray,
    eta: np.ndarray,
    k_du: float,
    k_cu: float
) -> Dict:
    """
    Input:
        r_min_bps: (N,) float
        eta: (N,) float

    Output:
        du_cpu_required: (N,) float
        cu_cpu_required: (N,) float
    """
    du_cpu_required = k_du * r_min_bps * (1.0 + eta)
    cu_cpu_required = k_cu * r_min_bps * (1.0 + eta)

    return {
        "du_cpu_required": du_cpu_required.astype(np.float64),
        "cu_cpu_required": cu_cpu_required.astype(np.float64),
    }