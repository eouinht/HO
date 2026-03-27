from SimulationConfig import create_default_config, set_random_seed
from TopologyBuilder import build_topology
from UEPositionGenerator import init_ue_state
from RadioSignalEstimator import estimate_radio_state

from ResourceStateManager import init_resource_state

import numpy as np 
from sim_step import run_single_simulation_step

cfg = create_default_config()
set_random_seed(cfg)

topology = build_topology(
    n_ru=cfg.n_ru,
    n_du=cfg.n_du,
    n_cu=cfg.n_cu,
    ru_prb_cap=cfg.ru_prb_cap,
    du_cpu_cap=cfg.du_cpu_cap,
    cu_cpu_cap=cfg.cu_cpu_cap,
)

# UE state
ue_pos, ue_vel, ue_slice = init_ue_state(
    n_ue=cfg.n_ue,
    speed_mean=cfg.ue_speed_mean,
    speed_std=cfg.ue_speed_std,
    area_size=500.0,
    embb_ratio=0.7,
)

# Radio State 
radio_state = estimate_radio_state(
    ue_pos=ue_pos,
    ru_pos=topology["ru_pos"],
    carrier_freq_ghz=cfg.carrier_freq_ghz,
    rb_bandwidth_hz=cfg.rb_bandwidth_hz,
    noise_figure_db=cfg.noise_figure_db,
    ru_tx_power_dbm=cfg.ru_tx_power_dbm,
    n_antennas=32,
)
resource_state = init_resource_state(
    serving_ru=radio_state["serving_ru"],
    prb_total=cfg.prb_total,
    ru_prb_cap=cfg.ru_prb_cap,
    n_ru=cfg.n_ru,
    total_tx_power_w=10 ** ((cfg.ru_tx_power_dbm - 30.0) / 10.0),
)

import numpy as np
queue_bits = np.zeros(cfg.n_ue, dtype=np.float64)

step_state = run_single_simulation_step(
    cfg=cfg,
    topology=topology,
    ue_pos=ue_pos,
    ue_vel=ue_vel,
    ue_slice=ue_slice,
    queue_bits=queue_bits,
    resource_state=resource_state,
)

print("candidate count:", step_state["filter_state"]["candidate_mask"].sum())
print("stable count:", step_state["filter_state"]["stable_mask"].sum())
print("qos violation count:", step_state["qos_violation"].sum())
print("throughput first 5:", step_state["traffic_state"]["throughput_bps"][:5])
print("latency first 5:", step_state["latency_state"]["total_latency_s"][:5])
print("required prb first 5:", step_state["ue_required_prb"][:5])
print("ru allocated prb:", step_state["resource_state"]["ru_prb_allocated"])
print("prb pool free:", step_state["resource_state"]["prb_pool_free"])