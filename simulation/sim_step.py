import numpy as np

from SimulationConfig import get_slice_params
from UEPositionGenerator import update_ue_positions
from RadioSignalEstimator import estimate_radio_state
from ResourceStateManager import (
    compute_ru_used_prb,
    compute_ru_free_prb,
    release_unused_prb,
    estimate_required_prb,
    compact_stable_ue_allocation,
    get_serving_du_cu,
    estimate_cpu_requirements,
)
from TrafficQueueManager import (
    estimate_traffic_state,
    compute_arrival_rate_packets_per_s,
    check_qos_violation,
)
from LatencyModel import estimate_latency_state
from simulation.HandoverCandidateFilter import classify_stable_and_candidate_ue


def run_single_simulation_step(
    cfg,
    topology,
    ue_pos,
    ue_vel,
    ue_slice,
    queue_bits,
    resource_state,
    area_size: float = 500.0,
):
    """
    One simulator step.

    Input:
        cfg: SimulationConfig
        topology: dict
        ue_pos: (N, 2)
        ue_vel: (N, 2)
        ue_slice: (N,)
        queue_bits: (N,)
        resource_state: dict

    Output:
        step_state: dict
    """
    # 1. mobility
    ue_pos_next, ue_vel_next = update_ue_positions(
        ue_pos=ue_pos,
        ue_vel=ue_vel,
        dt=cfg.time_step_s,
        area_size=area_size,
    )

    # 2. radio
    radio_state = estimate_radio_state(
        ue_pos=ue_pos_next,
        ru_pos=topology["ru_pos"],
        carrier_freq_ghz=cfg.carrier_freq_ghz,
        rb_bandwidth_hz=cfg.rb_bandwidth_hz,
        noise_figure_db=cfg.noise_figure_db,
        ru_tx_power_dbm=cfg.ru_tx_power_dbm,
        n_antennas=32,
    )

    serving_ru = radio_state["serving_ru"]
    ue_idx = np.arange(cfg.n_ue)
    serving_distance_m = radio_state["distance_m"][ue_idx, serving_ru]

    # 3. slice params
    r_min_bps, sinr_min_db, delay_max_s, eta, lambda_arrival_bps = get_slice_params(
        cfg=cfg,
        ue_slice=ue_slice,
    )

    packet_size_bits = np.where(
        ue_slice == 0,
        12000.0,
        4000.0,
    ).astype(np.float64)

    # 4. traffic from current resource allocation
    traffic_state = estimate_traffic_state(
        serving_gain=radio_state["serving_gain"],
        ue_power_alloc_w=resource_state["ue_power_alloc_w"],
        ue_allocated_prb=resource_state["ue_allocated_prb"],
        rb_bandwidth_hz=cfg.rb_bandwidth_hz,
        queue_bits=queue_bits,
        lambda_arrival_bps=lambda_arrival_bps,
        dt=cfg.time_step_s,
    )

    throughput_bps = traffic_state["throughput_bps"]
    queue_bits_next = traffic_state["queue_bits_next"]

    # 5. path RU->DU->CU
    path_state = get_serving_du_cu(
        serving_ru=serving_ru,
        ru_to_du=topology["ru_to_du"],
        du_to_cu=topology["du_to_cu"],
    )
    serving_du = path_state["serving_du"]
    serving_cu = path_state["serving_cu"]

    cpu_state = estimate_cpu_requirements(
        r_min_bps=r_min_bps,
        eta=eta,
        k_du=cfg.k_du,
        k_cu=cfg.k_cu,
    )

    du_cpu_capacity = topology["du_cpu_cap"][serving_du]
    cu_cpu_capacity = topology["cu_cpu_cap"][serving_cu]

    arrival_rate_packets_per_s = compute_arrival_rate_packets_per_s(
        lambda_arrival_bps=lambda_arrival_bps,
        packet_size_bits=packet_size_bits,
    )

    du_service_rate_packets_per_s = np.maximum(
        100.0 * du_cpu_capacity / np.maximum(np.mean(du_cpu_capacity), 1e-9),
        arrival_rate_packets_per_s + 1.0,
    )
    cu_service_rate_packets_per_s = np.maximum(
        100.0 * cu_cpu_capacity / np.maximum(np.mean(cu_cpu_capacity), 1e-9),
        arrival_rate_packets_per_s + 1.0,
    )

    latency_state = estimate_latency_state(
        serving_distance_m=serving_distance_m,
        packet_size_bits=packet_size_bits,
        throughput_bps=throughput_bps,
        arrival_rate_packets_per_s=arrival_rate_packets_per_s,
        du_cpu_required=cpu_state["du_cpu_required"],
        du_cpu_capacity=du_cpu_capacity,
        cu_cpu_required=cpu_state["cu_cpu_required"],
        cu_cpu_capacity=cu_cpu_capacity,
        du_service_rate_packets_per_s=du_service_rate_packets_per_s,
        cu_service_rate_packets_per_s=cu_service_rate_packets_per_s,
    )

    total_latency_s = latency_state["total_latency_s"]

    # 6. estimate required PRB
    ue_required_prb = estimate_required_prb(
        r_min_bps=r_min_bps,
        serving_gain=radio_state["serving_gain"],
        ue_power_alloc_w=resource_state["ue_power_alloc_w"],
        rb_bandwidth_hz=cfg.rb_bandwidth_hz,
    )

    # 7. filter stable / candidate
    filter_state = classify_stable_and_candidate_ue(
        serving_gain=radio_state["serving_gain"],
        best_neighbor_gain=radio_state["best_neighbor_gain"],
        throughput_bps=throughput_bps,
        total_latency_s=total_latency_s,
        r_min_bps=r_min_bps,
        delay_max_s=delay_max_s,
        ue_required_prb=ue_required_prb,
        current_allocated_prb=resource_state["ue_allocated_prb"],
    )

    stable_mask = filter_state["stable_mask"]
    candidate_mask = filter_state["candidate_mask"]

    # 8. compact stable UEs
    stable_margin_ratio = np.where(ue_slice == 0, 0.05, 0.15).astype(np.float64)

    compact_state = compact_stable_ue_allocation(
        serving_ru=serving_ru,
        stable_mask=stable_mask,
        ue_allocated_prb=resource_state["ue_allocated_prb"],
        ue_required_prb=ue_required_prb,
        stable_margin_ratio=stable_margin_ratio,
        n_ru=cfg.n_ru,
    )

    ue_allocated_prb_next = compact_state["ue_allocated_prb"]
    ru_used_prb_next = compact_state["ru_used_prb"]

    ru_free_prb_next = compute_ru_free_prb(
        ru_prb_allocated=resource_state["ru_prb_allocated"],
        ru_used_prb=ru_used_prb_next,
    )

    release_state = release_unused_prb(
        ru_prb_allocated=resource_state["ru_prb_allocated"],
        ru_used_prb=ru_used_prb_next,
        prb_pool_free=resource_state["prb_pool_free"],
    )

    qos_violation = check_qos_violation(
        throughput_bps=throughput_bps,
        total_latency_s=total_latency_s,
        r_min_bps=r_min_bps,
        delay_max_s=delay_max_s,
    )

    resource_state_next = {
        "ru_prb_allocated": release_state["ru_prb_allocated"],
        "ru_used_prb": ru_used_prb_next,
        "ru_free_prb": ru_free_prb_next,
        "ue_allocated_prb": ue_allocated_prb_next,
        "ue_power_alloc_w": resource_state["ue_power_alloc_w"],
        "prb_pool_free": release_state["prb_pool_free"],
        "ue_count_per_ru": np.bincount(serving_ru, minlength=cfg.n_ru).astype(np.float64),
    }

    return {
        "ue_pos": ue_pos_next,
        "ue_vel": ue_vel_next,
        "queue_bits": queue_bits_next,
        "radio_state": radio_state,
        "traffic_state": traffic_state,
        "latency_state": latency_state,
        "filter_state": filter_state,
        "resource_state": resource_state_next,
        "qos_violation": qos_violation,
        "ue_required_prb": ue_required_prb,
    }