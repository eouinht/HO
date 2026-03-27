import numpy as np
from typing import Dict

def compute_snr_per_rb(
    serving_gain: np.ndarray,
    ue_power_alloc_w: np.ndarray,
    ue_allocated_prb: np.ndarray
)-> np.ndarray:
    """
    Input:
        serving_gain: (N,) float
        ue_power_alloc_w: (N,) float
        ue_allocated_prb: (N,) float

    Output:
        snr_per_rb: (N,) float
    """
    power_per_rb = ue_power_alloc_w/np.maximum(ue_allocated_prb, 1e-9)
    snr_per_rb = power_per_rb*serving_gain
    return np.maximum(snr_per_rb, 0.0).astype(np.float64)

def estimate_ue_throughput_bps(
    serving_gain: np.ndarray,
    ue_power_alloc_w: np.ndarray,
    ue_allocated_prb: np.ndarray,
    rb_bandwidth_hz: float
) -> np.ndarray:
    """
    Throughput model from network_env.py logic:
        snr_per_rb = (p_k / N_RB) * gain
        R_k = N_RB * BW_RB * log2(1 + snr_per_rb)

    Input:
        serving_gain: (N,) float
        ue_power_alloc_w: (N,) float
        ue_allocated_prb: (N,) float
        rb_bandwidth_hz: float

    Output:
        throughput_bps: (N,) float
    """
    snr_per_rb = compute_snr_per_rb(
        serving_gain=serving_gain,
        ue_power_alloc_w=ue_power_alloc_w,
        ue_allocated_prb=ue_allocated_prb,
    )
    throughput_bps = ue_allocated_prb * rb_bandwidth_hz * np.log2(1.0 + snr_per_rb)
    return np.maximum(throughput_bps, 0.0).astype(np.float64)


def generate_arrival_bits(
    lambda_arrival_bps: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Input:
        lambda_arrival_bps: (N,) float
        dt: float

    Output:
        arrival_bits: (N,) float
    """
    arrival_bits = lambda_arrival_bps * dt
    return np.maximum(arrival_bits, 0.0).astype(np.float64)


def update_queue_bits(
    queue_bits: np.ndarray,
    arrival_bits: np.ndarray,
    served_bits: np.ndarray
) -> np.ndarray:
    """
    Input:
        queue_bits: (N,) float
        arrival_bits: (N,) float
        served_bits: (N,) float

    Output:
        queue_bits_next: (N,) float
    """
    queue_bits_next = queue_bits + arrival_bits - served_bits
    return np.maximum(queue_bits_next, 0.0).astype(np.float64)


def compute_served_bits(
    throughput_bps: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Input:
        throughput_bps: (N,) float
        dt: float

    Output:
        served_bits: (N,) float
    """
    served_bits = throughput_bps * dt
    return np.maximum(served_bits, 0.0).astype(np.float64)


def compute_arrival_rate_packets_per_s(
    lambda_arrival_bps: np.ndarray,
    packet_size_bits: np.ndarray
) -> np.ndarray:
    """
    Input:
        lambda_arrival_bps: (N,) float
        packet_size_bits: (N,) float

    Output:
        arrival_rate_packets_per_s: (N,) float
    """
    arrival_rate_packets_per_s = lambda_arrival_bps / np.maximum(packet_size_bits, 1e-9)
    return np.maximum(arrival_rate_packets_per_s, 0.0).astype(np.float64)


def check_qos_violation(
    throughput_bps: np.ndarray,
    total_latency_s: np.ndarray,
    r_min_bps: np.ndarray,
    delay_max_s: np.ndarray
) -> np.ndarray:
    """
    Input:
        throughput_bps: (N,) float
        total_latency_s: (N,) float
        r_min_bps: (N,) float
        delay_max_s: (N,) float

    Output:
        qos_violation: (N,) bool
    """
    throughput_violation = throughput_bps < r_min_bps
    latency_violation = total_latency_s > delay_max_s
    return (throughput_violation | latency_violation).astype(bool)


def estimate_traffic_state(
    serving_gain: np.ndarray,
    ue_power_alloc_w: np.ndarray,
    ue_allocated_prb: np.ndarray,
    rb_bandwidth_hz: float,
    queue_bits: np.ndarray,
    lambda_arrival_bps: np.ndarray,
    dt: float
) -> Dict:
    """
    Input:
        serving_gain: (N,) float
        ue_power_alloc_w: (N,) float
        ue_allocated_prb: (N,) float
        rb_bandwidth_hz: float
        queue_bits: (N,) float
        lambda_arrival_bps: (N,) float
        dt: float

    Output:
        traffic_state: dict
            throughput_bps: (N,)
            arrival_bits: (N,)
            served_bits: (N,)
            queue_bits_next: (N,)
    """
    throughput_bps = estimate_ue_throughput_bps(
        serving_gain=serving_gain,
        ue_power_alloc_w=ue_power_alloc_w,
        ue_allocated_prb=ue_allocated_prb,
        rb_bandwidth_hz=rb_bandwidth_hz,
    )

    arrival_bits = generate_arrival_bits(
        lambda_arrival_bps=lambda_arrival_bps,
        dt=dt,
    )

    served_bits = compute_served_bits(
        throughput_bps=throughput_bps,
        dt=dt,
    )

    queue_bits_next = update_queue_bits(
        queue_bits=queue_bits,
        arrival_bits=arrival_bits,
        served_bits=served_bits,
    )

    return {
        "throughput_bps": throughput_bps,
        "arrival_bits": arrival_bits,
        "served_bits": served_bits,
        "queue_bits_next": queue_bits_next,
    }