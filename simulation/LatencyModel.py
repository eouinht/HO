import numpy as np 
from typing import Dict 

def compute_propagation_delay_s(
    distance_m: np.ndarray,
    propagation_speed_mps: float=3.0e8
)-> np.ndarray:
    """
    Input:
        distance_m: (N,) float
        propagation_speed_mps: float

    Output:
        propagation_delay_s: (N,) float
    """
    
    propagation_delay_s = distance_m/propagation_speed_mps
    return np.maximum(propagation_delay_s, 0.0).astype(np.float64)

def compute_transmission_delay_s(
    packet_size_bits: np.ndarray,
    throughput_bps: np.ndarray
)-> np.ndarray:
    """
    Input:
        packet_size_bits: (N,) float
        throughput_bps: (N,) float

    Output:
        transmission_delay_s: (N,) float
    """
    transmission_delay_s = packet_size_bits/np.maximum(throughput_bps, 1e-9)
    return np.maximum(transmission_delay_s, 0.0).astype(np.float64)

def compute_processing_delay_du_s(
    du_cpu_required: np.ndarray,
    du_cpu_capacity: np.ndarray,
    base_processing_delay_s: float= 1e-4
)-> np.ndarray:
    """
    DU processing delay surrogate.

    Input:
        du_cpu_required: (N,) float
        du_cpu_capacity: (N,) float

    Output:
        processing_delay_du_s: (N,) float
    """
    load_ratio = du_cpu_required/np.maximum(du_cpu_capacity, 1e-9)
    processing_delay_du_s = base_processing_delay_s*(1.0 + load_ratio)
    return np.maximum(processing_delay_du_s, 0.0).astype(np.float64)

def compute_processing_delay_cu_s(
    cu_cpu_required: np.ndarray,
    cu_cpu_capacity: np.ndarray,
    base_processing_delay_s: float= 1e-4
    
)-> np.ndarray:
    """
    CU processing delay surrogate.

    Input:
        cu_cpu_required: (N,) float
        cu_cpu_capacity: (N,) float

    Output:
        processing_delay_cu_s: (N,) float
    """
    load_ratio = cu_cpu_required/np.maximum(cu_cpu_capacity, 1e-9)
    processing_delay_cu_s = base_processing_delay_s*(1.0 + load_ratio)
    return np.maximum(processing_delay_cu_s, 0.0).astype(np.float64)

def compute_nm1_queue_delay_s(
    arrival_rate: np.ndarray,
    service_rate: np.ndarray,
    min_stable_gap: float = 1e-9
)-> np.ndarray:
    """
    M/M/1 queue delay surrogate:
        W = 1 / (mu - lambda), if mu > lambda
    If unstable, returns a large penalty delay.

    Input:
        arrival_rate: (N,) float
        service_rate: (N,) float

    Output:
        queue_delay_s: (N,) float
    """
    
    gap = service_rate - arrival_rate
    stable_mask = gap > min_stable_gap
    
    queue_delay_s = np.full(arrival_rate.shape, 1.0, dtype=np.float64)
    queue_delay_s[stable_mask] = 1.0/gap[stable_mask]
    return np.maximum(queue_delay_s, 0.0).astype(np.float64)

def compute_du_queue_delay_s(
    arrival_rate: np.ndarray,
    du_service_rate: np.ndarray
)-> np.ndarray:
    """
    Input:
        arrival_rate: (N,) float
        du_service_rate: (N,) float

    Output:
        queue_delay_du_s: (N,) float
    """
    return compute_nm1_queue_delay_s(arrival_rate, du_service_rate)

def compute_cu_queue_delay_s(
    arrival_rate: np.ndarray,
    cu_service_rate: np.ndarray
)-> np.ndarray:
    """
    Input:
        arrival_rate: (N,) float
        cu_service_rate: (N,) float

    Output:
        queue_delay_cu_s: (N,) float
    """
    return compute_nm1_queue_delay_s(arrival_rate, cu_service_rate)

def compute_total_latency_s(
    propagation_delay_s: np.ndarray,
    transmission_delay_s: np.ndarray,
    processing_delay_du_s: np.ndarray,
    processing_delay_cu_s: np.ndarray,
    queue_delay_du_s: np.ndarray,
    queue_delay_cu_s: np.ndarray,
    ho_delay_s:np.ndarray
)-> np.ndarray:
    """
    Input:
        propagation_delay_s: (N,) float
        transmission_delay_s: (N,) float
        processing_delay_du_s: (N,) float
        processing_delay_cu_s: (N,) float
        queue_delay_du_s: (N,) float
        queue_delay_cu_s: (N,) float
        ho_delay_s: (N,) float or None

    Output:
        total_latency_s: (N,) float
    """
    total_latency_s = (
        propagation_delay_s
        + transmission_delay_s
        + processing_delay_du_s
        + processing_delay_cu_s
        + queue_delay_du_s
        + queue_delay_cu_s
    )

    if ho_delay_s is not None:
        total_latency_s = total_latency_s + ho_delay_s

    return np.maximum(total_latency_s, 0.0).astype(np.float64)

def estimate_latency_state(
    serving_distance_m: np.ndarray,
    packet_size_bits: np.ndarray,
    throughput_bps: np.ndarray,
    arrival_rate_packets_per_s: np.ndarray,
    du_cpu_required: np.ndarray,
    du_cpu_capacity: np.ndarray,
    cu_cpu_required: np.ndarray,
    cu_cpu_capacity: np.ndarray,
    du_service_rate_packets_per_s: np.ndarray,
    cu_service_rate_packets_per_s: np.ndarray,
    ho_delay_s: np.ndarray = None,
) -> Dict:
    """
    Input:
        serving_distance_m: (N,) float
        packet_size_bits: (N,) float
        throughput_bps: (N,) float
        arrival_rate_packets_per_s: (N,) float
        du_cpu_required: (N,) float
        du_cpu_capacity: (N,) float
        cu_cpu_required: (N,) float
        cu_cpu_capacity: (N,) float
        du_service_rate_packets_per_s: (N,) float
        cu_service_rate_packets_per_s: (N,) float
        ho_delay_s: (N,) float or None

    Output:
        latency_state: dict
            propagation_delay_s: (N,)
            transmission_delay_s: (N,)
            processing_delay_du_s: (N,)
            processing_delay_cu_s: (N,)
            queue_delay_du_s: (N,)
            queue_delay_cu_s: (N,)
            total_latency_s: (N,)
    """
    propagation_delay_s = compute_propagation_delay_s(
        distance_m=serving_distance_m,
    )

    transmission_delay_s = compute_transmission_delay_s(
        packet_size_bits=packet_size_bits,
        throughput_bps=throughput_bps,
    )

    processing_delay_du_s = compute_processing_delay_du_s(
        du_cpu_required=du_cpu_required,
        du_cpu_capacity=du_cpu_capacity,
    )

    processing_delay_cu_s = compute_processing_delay_cu_s(
        cu_cpu_required=cu_cpu_required,
        cu_cpu_capacity=cu_cpu_capacity,
    )

    queue_delay_du_s = compute_du_queue_delay_s(
        arrival_rate=arrival_rate_packets_per_s,
        du_service_rate=du_service_rate_packets_per_s,
    )

    queue_delay_cu_s = compute_cu_queue_delay_s(
        arrival_rate=arrival_rate_packets_per_s,
        cu_service_rate=cu_service_rate_packets_per_s,
    )

    total_latency_s = compute_total_latency_s(
        propagation_delay_s=propagation_delay_s,
        transmission_delay_s=transmission_delay_s,
        processing_delay_du_s=processing_delay_du_s,
        processing_delay_cu_s=processing_delay_cu_s,
        queue_delay_du_s=queue_delay_du_s,
        queue_delay_cu_s=queue_delay_cu_s,
        ho_delay_s=ho_delay_s,
    )

    return {
        "propagation_delay_s": propagation_delay_s,
        "transmission_delay_s": transmission_delay_s,
        "processing_delay_du_s": processing_delay_du_s,
        "processing_delay_cu_s": processing_delay_cu_s,
        "queue_delay_du_s": queue_delay_du_s,
        "queue_delay_cu_s": queue_delay_cu_s,
        "total_latency_s": total_latency_s,
    }