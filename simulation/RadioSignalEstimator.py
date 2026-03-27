import numpy as np
from typing import Tuple, Dict 

def compute_distance_matrix(ue_pos: np.ndarray, ru_pos: np.ndarray) -> np.ndarray:
    """
    Input:
        ue_pos: (N, 2) float
        ru_pos: (M, 2) float

    Output:
        distance_m: (N, M) float
    """
    diff = ue_pos[:, None, :] - ru_pos[None, :, :]
    distance_m = np.linalg.norm(diff, axis=2)
    return np.maximum(distance_m, 1.0).astype(np.float64)

def compute_pathloss_db(distance_m: np.ndarray, carrier_freq_ghz: float) -> np.ndarray:
    """
    Simplified 3GPP UMa-like pathloss surrogate.

    Input:
        distance_m: (N, M) float
        carrier_freq_ghz: float

    Output:
        pathloss_db: (N, M) float
    """
    
    pathloss_db = (
        32.4
        + 21.0 * np.log10(distance_m)
        + 20.0 * np.log10(carrier_freq_ghz)
    )
    return pathloss_db.astype(np.float64)

def compute_noise_power_per_rb_w(rb_bandwidth_hz: float, noise_figure_db: float)-> float:
    """
    Simplified 3GPP UMa-like pathloss surrogate.

    Input:
        distance_m: (N, M) float
        carrier_freq_ghz: float

    Output:
        pathloss_db: (N, M) float
    """
    noise_power_dbm = -174.0 + 10.0 * np.log10(rb_bandwidth_hz) + noise_figure_db
    noise_power_w = 10.0 ** ((noise_power_dbm - 30.0) / 10.0)
    return float(noise_power_w)


def compute_rsrp_dbm(pathloss_db: np.ndarray, ru_tx_power_dbm: float) -> np.ndarray:
    """
    Input:
        pathloss_db: (N, M) float
        ru_tx_power_dbm: float

    Output:
        rsrp_dbm: (N, M) float
    """
    rsrp_dbm = ru_tx_power_dbm - pathloss_db
    return rsrp_dbm.astype(np.float64)

def compute_noise_power_dbm(bandwidth_mhz: float, noise_figure_db: float) -> float:
    """
    Thermal noise approximation:
        noise_dbm = -174 + 10*log10(B_hz) + NF

    Input:
        bandwidth_mhz: float
        noise_figure_db: float

    Output:
        noise_power_dbm: float
    """
    bandwidth_hz = bandwidth_mhz * 1e6
    noise_power_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz) + noise_figure_db
    return float(noise_power_dbm)


def dbm_to_w(power_dbm: np.ndarray) -> np.ndarray:
    """
    Input:
        power_dbm: ndarray

    Output:
        power_w: ndarray
    """
    return np.power(10.0, (power_dbm - 30.0)/ 10.0)


def w_to_dbm(power_w: np.ndarray) -> np.ndarray:
    """
    Input:
        power_w: ndarray

    Output:
        power_dbm: ndarray
    """
    return 10.0 * np.log10(np.maximum(power_w, 1e-12)) + 30.0

def generate_rayleigh_channel_power(n_ue: int, n_ru: int, n_antennas: int=32)-> np.ndarray:
    """
    Generate Rayleigh fading channel power.

    Input:
        n_ue: int
        n_ru: int
        n_antennas: int

    Output:
        channel_power: (N, M) float
    """
    real = np.random.randn(n_ue, n_ru, n_antennas)
    imag = np.random.randn(n_ue, n_ru, n_antennas)
    h = (real + 1j * imag) / np.sqrt(2.0)
    channal_power = np.sum(np.abs(h) **2, axis=2)
    return channal_power.astype(np.float64)

def compute_large_scale_power_w(
    ru_tx_power_dbm:float,
    pathloss_db: np.ndarray
)-> np.ndarray:
    """
    Generate Rayleigh fading channel power.

    Input:
        n_ue: int
        n_ru: int
        n_antennas: int

    Output:
        channel_power: (N, M) float
    """
    rx_power_dbm = ru_tx_power_dbm - pathloss_db
    return dbm_to_w(rx_power_dbm).astype(np.float64)

def compute_channel_power_w(
    large_scale_power_w: np.ndarray,
    fading_power: np.ndarray,
    n_antennas: int
)-> np.ndarray:
    """
    Received large-scale power in Watt without fading.

    Input:
        ru_tx_power_dbm: float
        pathloss_db: (N, M) float

    Output:
        large_scale_power_w: (N, M) float
    """
    normalized_fading= fading_power/float(n_antennas)
    channel_power_w = large_scale_power_w*normalized_fading
    return channel_power_w.astype(np.float64)

def compute_gain( channel_power_w: np.ndarray,
                 noise_power_rb_w: float) -> np.ndarray:
    """
    Apply Rayleigh fading to large-scale received power.

    Input:
        large_scale_power_w: (N, M) float
        fading_power: (N, M) float
        n_antennas: int

    Output:
        channel_power_w: (N, M) float
    """
    gain = channel_power_w/max(noise_power_rb_w, 1e-30)
    return gain.astype(np.float64)



def compute_rsrp_dbm( channel_power_w: np.ndarray) -> np.ndarray:
    """
    Approximate RSRP from channel power.

    Input:
        channel_power_w: (N, M) float

    Output:
        rsrp_dbm: (N, M) float
    """
    return w_to_dbm(channel_power_w).astype(np.float64)


def select_serving_ru_from_rsrp( rsrp_dbm: np.ndarray) -> np.ndarray:
    """
    Input:
        rsrp_dbm: (N, M) float

    Output:
        serving_ru: (N,) int
    """
    return np.argmax(rsrp_dbm, axis=1).astype(np.int32)


def extract_serving_gain( gain: np.ndarray, serving_ru: np.ndarray ) -> np.ndarray:
    """
    Input:
        gain: (N, M) float
        serving_ru: (N,) int

    Output:
        serving_gain: (N,) float
    """
    ue_idx = np.arange(serving_ru.shape[0])
    return gain[ue_idx, serving_ru].astype(np.float64)


def extract_best_neighbor_ru( rsrp_dbm: np.ndarray, serving_ru: np.ndarray) -> np.ndarray:
    """
    Input:
        rsrp_dbm: (N, M) float
        serving_ru: (N,) int

    Output:
        best_neighbor_ru: (N,) int
    """
    rsrp_copy = rsrp_dbm.copy()
    ue_idx = np.arange(serving_ru.shape[0])
    rsrp_copy[ue_idx, serving_ru] = -1e12
    return np.argmax(rsrp_copy, axis=1).astype(np.int32)


def extract_best_neighbor_gain( gain: np.ndarray, best_neighbor_ru: np.ndarray) -> np.ndarray:
    """
    Input:
        gain: (N, M) float
        best_neighbor_ru: (N,) int

    Output:
        best_neighbor_gain: (N,) float
    """
    ue_idx = np.arange(best_neighbor_ru.shape[0])
    return gain[ue_idx, best_neighbor_ru].astype(np.float64)


def estimate_radio_state(
    ue_pos: np.ndarray,
    ru_pos: np.ndarray,
    carrier_freq_ghz: float,
    rb_bandwidth_hz: float,
    noise_figure_db: float,
    ru_tx_power_dbm: float,
    n_antennas: int = 32,
) -> Dict:
    """
    Input:
        ue_pos: (N, 2) float
        ru_pos: (M, 2) float
        carrier_freq_ghz: float
        rb_bandwidth_hz: float
        noise_figure_db: float
        ru_tx_power_dbm: float
        n_antennas: int

    Output:
        radio_state: dict
            distance_m: (N, M)
            pathloss_db: (N, M)
            noise_power_rb_w: float
            fading_power: (N, M)
            channel_power_w: (N, M)
            gain: (N, M)
            rsrp_dbm: (N, M)
            serving_ru: (N,)
            serving_gain: (N,)
            best_neighbor_ru: (N,)
            best_neighbor_gain: (N,)
    """
    n_ue = ue_pos.shape[0]
    n_ru = ru_pos.shape[0]

    distance_m = compute_distance_matrix(
        ue_pos=ue_pos,
        ru_pos=ru_pos,
    )

    pathloss_db = compute_pathloss_db(
        distance_m=distance_m,
        carrier_freq_ghz=carrier_freq_ghz,
    )

    noise_power_rb_w = compute_noise_power_per_rb_w(
        rb_bandwidth_hz=rb_bandwidth_hz,
        noise_figure_db=noise_figure_db,
    )

    fading_power = generate_rayleigh_channel_power(
        n_ue=n_ue,
        n_ru=n_ru,
        n_antennas=n_antennas,
    )

    large_scale_power_w = compute_large_scale_power_w(
        ru_tx_power_dbm=ru_tx_power_dbm,
        pathloss_db=pathloss_db,
    )

    channel_power_w = compute_channel_power_w(
        large_scale_power_w=large_scale_power_w,
        fading_power=fading_power,
        n_antennas=n_antennas,
    )

    gain = compute_gain(
        channel_power_w=channel_power_w,
        noise_power_rb_w=noise_power_rb_w,
    )

    rsrp_dbm = compute_rsrp_dbm(
        channel_power_w=channel_power_w,
    )

    serving_ru = select_serving_ru_from_rsrp(
        rsrp_dbm=rsrp_dbm,
    )

    serving_gain = extract_serving_gain(
        gain=gain,
        serving_ru=serving_ru,
    )

    best_neighbor_ru = extract_best_neighbor_ru(
        rsrp_dbm=rsrp_dbm,
        serving_ru=serving_ru,
    )

    best_neighbor_gain = extract_best_neighbor_gain(
        gain=gain,
        best_neighbor_ru=best_neighbor_ru,
    )

    return {
        "distance_m": distance_m,
        "pathloss_db": pathloss_db,
        "noise_power_rb_w": noise_power_rb_w,
        "fading_power": fading_power,
        "channel_power_w": channel_power_w,
        "gain": gain,
        "rsrp_dbm": rsrp_dbm,
        "serving_ru": serving_ru,
        "serving_gain": serving_gain,
        "best_neighbor_ru": best_neighbor_ru,
        "best_neighbor_gain": best_neighbor_gain,
    }