import numpy as np
from typing import Dict


def classify_stable_and_candidate_ue(
    serving_gain: np.ndarray,
    best_neighbor_gain: np.ndarray,
    throughput_bps: np.ndarray,
    total_latency_s: np.ndarray,
    r_min_bps: np.ndarray,
    delay_max_s: np.ndarray,
    ue_required_prb: np.ndarray,
    current_allocated_prb: np.ndarray,
    min_gain_improvement_ratio: float = 1.10,
    prb_saving_ratio_threshold: float = 0.10
) -> Dict:
    """
    UE is candidate if:
    1) neighbor gain is sufficiently better than serving gain
    OR
    2) QoS is violated
    OR
    3) current PRB is much larger than required PRB (resource inefficiency)

    Input:
        serving_gain: (N,) float
        best_neighbor_gain: (N,) float
        throughput_bps: (N,) float
        total_latency_s: (N,) float
        r_min_bps: (N,) float
        delay_max_s: (N,) float
        ue_required_prb: (N,) float
        current_allocated_prb: (N,) float

    Output:
        result: dict
            stable_mask: (N,) bool
            candidate_mask: (N,) bool
            gain_improvement_mask: (N,) bool
            qos_violation_mask: (N,) bool
            resource_inefficiency_mask: (N,) bool
    """
    gain_improvement_mask = best_neighbor_gain >= serving_gain * min_gain_improvement_ratio

    qos_violation_mask = (throughput_bps < r_min_bps) | (total_latency_s > delay_max_s)

    prb_saving_ratio = 1.0 - (ue_required_prb / np.maximum(current_allocated_prb, 1e-9))
    resource_inefficiency_mask = prb_saving_ratio >= prb_saving_ratio_threshold

    candidate_mask = gain_improvement_mask | qos_violation_mask | resource_inefficiency_mask
    stable_mask = ~candidate_mask

    return {
        "stable_mask": stable_mask.astype(bool),
        "candidate_mask": candidate_mask.astype(bool),
        "gain_improvement_mask": gain_improvement_mask.astype(bool),
        "qos_violation_mask": qos_violation_mask.astype(bool),
        "resource_inefficiency_mask": resource_inefficiency_mask.astype(bool)
    }