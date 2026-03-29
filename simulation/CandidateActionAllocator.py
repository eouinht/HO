import numpy as np
from typing import Dict

from .HandoverFeasibleChecker import check_handover_feasibility
def get_priority(
    qos_violation_mask: np.ndarray,
    radio_better: np.ndarray,
    prb_waste_mask: np.ndarray
)-> np.ndarray:
    """
    Returns:
        priority: (N,) int
            2 = highest (QoS)
            1 = radio
            0 = resource
           -1 = not candidate
    """

    priority = np.full(qos_violation_mask.shape, -1, dtype=np.int32)

    # Level 1: QoS violation
    priority[qos_violation_mask] = 2

    # Level 2: radio better (but not QoS)
    mask_radio = radio_better & (~qos_violation_mask)
    priority[mask_radio] = 1

    # Level 3: resource waste only
    mask_resource = prb_waste_mask & (~qos_violation_mask) & (~radio_better)
    priority[mask_resource] = 0

    return priority


def sort_candidate(
    candidate_mask: np.ndarray,
    qos_violation_mask: np.ndarray,
    radio_better: np.ndarray,
    prb_waste_mask: np.ndarray
)-> np.ndarray:
    """
    Input:
        candidate_mask: (N,) bool
        qos_violation_mask: (N,) bool
        radio_better: (N,) bool
        prb_waste_mask: (N,) bool

    Output:
        sorted_candidate: (K,) int
    """
    candidate_idx = np.where(candidate_mask)[0]
    if candidate_idx.size == 0:
        return candidate_idx
    
    priority = get_priority(
        qos_violation_mask=qos_violation_mask,
        radio_better=radio_better,
        prb_waste_mask=prb_waste_mask,
    )
    
    order = np.argsort(-priority[candidate_idx])
    return candidate_idx[order]    

def apply_single_handover(
    ue_id: int,
    source_ru: int,
    target_ru: int,
    required_prb: float,
    resource_state: Dict,
    du_cpu_used: np.ndarray,
    cu_cpu_used: np.ndarray,
    du_cpu_required: float,
    cu_cpu_required: float,
    topology: Dict
)-> Dict:
    """
    Apply HO: update resource + CPU + serving

    Output:
        updated resource_state, du_cpu_used, cu_cpu_used
        
    """
    resource_state_new = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in resource_state.items()
    }
    du_cpu_used_new = du_cpu_used.copy()
    cu_cpu_used_new = cu_cpu_used.copy()


    old_prb = resource_state["ue_allocated_prb"][ue_id]
    
    # relese
    resource_state_new["ru_used_prb"] -= old_prb
    # allocated to target ru 
    resource_state_new["ue_allocated_prb"]["ue_id"] = required_prb
    resource_state_new["ru_used_prb"][target_ru] += required_prb
 
    # update free prb
    resource_state_new["ru_free_prb"] = (
        resource_state_new["ru_prb_allocated"] 
        - resource_state_new["ru_used_prb"]
        )
    
    # update cpu load
    source_du = topology["ru_to_du"][source_ru]
    source_cu = topology["du_to_cu"][source_du]
    
    target_du = topology["ru_to_du"][target_ru]
    target_cu = topology["du_to_cu"][target_du]
    
    du_cpu_used_new[source_du] -= du_cpu_required
    cu_cpu_used_new[source_cu] -= cu_cpu_required
    
    du_cpu_used_new[target_du] += du_cpu_required
    cu_cpu_used_new[target_cu] += cu_cpu_required
    
    return {
        "resource_state": resource_state_new,
        "du_cpu_used": du_cpu_used_new,
        "cu_cpu_used": cu_cpu_used_new,
    }
    
def process_candidate_ues(
    candidate_mask: np.ndarray,
    filter_state:Dict,
    radio_state: Dict,
    topology: Dict,
    resource_state: Dict,
    r_min_bps: np.ndarray,
    delay_max_s: np.ndarray,
    packet_size_bits: np.ndarray,
    lambda_arrival_bps: np.ndarray,
    du_cpu_required: np.ndarray,
    cu_cpu_required: np.ndarray,
    du_cpu_used: np.ndarray,
    cu_cpu_used: np.ndarray,
    rb_bandwidth_hz: float,
    ru_prb_cap: float,
)-> Dict:
    """
    Process candidate UEs in priority order:
        QoS violation > radio better > PRB waste

    Input:
        candidate_mask: (N,) bool
        filter_state: dict
            must contain:
                radio_better: (N,)
                qos_violation_mask: (N,)
                prb_waste_mask: (N,)
        radio_state: dict
        topology: dict
        resource_state: dict
        r_min_bps: (N,)
        delay_max_s: (N,)
        packet_size_bits: (N,)
        lambda_arrival_bps: (N,)
        du_cpu_required: (N,)
        cu_cpu_required: (N,)
        du_cpu_used: (D,)
        cu_cpu_used: (C,)
        rb_bandwidth_hz: float
        ru_prb_cap: float

    Output:
        result: dict
            serving_ru: (N,)
            resource_state: dict
            du_cpu_used: (D,)
            cu_cpu_used: (C,)
            ho_applied_mask: (N,) bool
            ho_result_code: (N,) int
                0: untouched/non-candidate
                1: applied
               -1: infeasible
               -2: target == source
    """
    
    serving_ru = radio_state["serving_ru"].copy()
    best_neighbor_ru = radio_state["best_neighbor_ru"]
    
    radio_better = filter_state["radio_better"]
    qos_violation_mask = filter_state["qos_violation_mask"]
    prb_waste_mask = filter_state["prb_waste_mask"]

    sorted_candidate_idx = sort_candidate(
        candidate_mask=candidate_mask,
        qos_violation_mask=qos_violation_mask,
        radio_better=radio_better,
        prb_waste_mask=prb_waste_mask,
    )
    
    ho_applied_mask = np.zeros(candidate_mask.shape, dtype=bool)
    ho_result_code = np.zeros(candidate_mask.shape, dtype=np.int32)

    resource_state_cur = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in resource_state.items()
    }
    du_cpu_used_cur = du_cpu_used.copy()
    cu_cpu_used_cur = cu_cpu_used.copy()

    for ue_id in sorted_candidate_idx:
        source_ru = int(serving_ru[ue_id])
        target_ru = int(best_neighbor_ru[ue_id])
        
        if target_ru == source_ru:
            continue
            
        result = check_handover_feasibility(
            ue_id=ue_id,
            source_ru=source_ru,
            target_ru=target_ru,
            radio_state=radio_state,
            topology=topology,
            resource_state=resource_state_cur,
            rb_bandwidth_hz=rb_bandwidth_hz,
            r_min_bps=r_min_bps,
            delay_max_s=delay_max_s,
            packet_size_bits=packet_size_bits,
            lambda_arrival_bps=lambda_arrival_bps,
            du_cpu_required=du_cpu_required,
            cu_cpu_required=cu_cpu_required,
            du_cpu_used=du_cpu_used_cur,
            cu_cpu_used=cu_cpu_used_cur,
            ru_prb_cap=ru_prb_cap
        )
        
        if not result["feasible"]:
            continue
        
        # === APPLY HO ===
        apply_out = apply_single_handover(
            ue_id=ue_id,
            source_ru=source_ru,
            target_ru=target_ru,
            required_prb=result["required_prb"],
            resource_state=resource_state_cur,
            du_cpu_used=du_cpu_used_cur,
            cu_cpu_used=cu_cpu_used_cur,
            du_cpu_required=du_cpu_required[ue_id],
            cu_cpu_required=cu_cpu_required[ue_id],
            topology=topology,
        )
        resource_state = apply_out["resource_state"]
        du_cpu_used = apply_out["du_cpu_used"]
        cu_cpu_used = apply_out["cu_cpu_used"]
        
        serving_ru[ue_id] = target_ru
        ho_applied_mask[ue_id] = True
        ho_result_code[ue_id] = 1
        
       
        
    return {
        "serving_ru": serving_ru,
        "resource_state": resource_state_cur,
        "du_cpu_used": du_cpu_used_cur,
        "cu_cpu_used": cu_cpu_used_cur,
        "ho_applied_mask": ho_applied_mask,
        "ho_result_code": ho_result_code,
        "sorted_candidate_idx": sorted_candidate_idx,
    }