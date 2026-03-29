import random
import numpy as np

from config import *
from Env.network_env import NetworkEnv


def build_env(num_ues=50, num_rbs=135, dynamic_mode=True):
    env = NetworkEnv(
        total_nodes=total_nodes,
        num_RUs=num_RUs,
        num_DUs=num_DUs,
        num_CUs=num_CUs,
        num_RBs=num_rbs,
        num_UEs=num_ues,
        SLICE_PRESET=SLICE_PRESET,
        P_i_random_list=P_i_random_list,
        A_j_random_list=A_j_random_list,
        A_m_random_list=A_m_random_list,
        bw_ru_du_random_list=bw_ru_du_random_list,
        bw_du_cu_random_list=bw_du_cu_random_list,
        bandwidth_per_RB=bandwidth_per_RB,
        max_RBs_per_UE=max_RBs_per_UE,
        P_ib_sk_val=P_ib_sk_val,
        k_DU=k_DU,
        k_CU=k_CU,
        dynamic_mode=dynamic_mode,
        min_ues=45,
        max_ues=55,
        mobility_step=20.0,
    )
    return env


def print_step_info(state, info):
    print(f"time_step        : {info['time_step']}")
    print(f"num_active_ues   : {state['num_active_ues']}")
    print(f"departed_ues     : {info['departed_ues']}")
    print(f"new_ue_ids       : {info['new_ue_ids']}")
    print(f"stable_ues       : {len(info['stable_ues'])}")
    print(f"ho_candidates    : {len(info['ho_candidates'])}")
    print(f"new_ue_candidates: {len(info['new_ue_candidates'])}")
    print(f"RB_remaining     : {state['RB_remaining']}")
    print(f"PRB_per_RU       : {state['PRB_remaining_per_RU']}")
    print("-" * 50)


def main():
    seed = 2
    random.seed(seed)
    np.random.seed(seed)

    env = build_env(num_ues=60, num_rbs=135, dynamic_mode=True)

    print("=== RESET ===")
    state = env.reset_env()
    print("num_active_ues:", state["num_active_ues"])
    print("RB_remaining :", state["RB_remaining"])
    print("PRB_per_RU   :", state["PRB_remaining_per_RU"])
    print("-" * 50)

    for t in range(5):
        target_active = int(np.random.randint(45, 56))
        print(f"=== ADVANCE {t+1} | target_active={target_active} ===")
        state, info = env.advance_time(target_active_ues=target_active)
        accepted = 0
        reasons = {}

        for ue_id in info["new_ue_candidates"]:
            ok, msg = env.check_feasible(
                UE_idx=ue_id,
                RU_choice=0,
                DU_choice=0,
                CU_choice=0,
                num_RB_alloc=1,
                power_level_alloc=env.P_ib_sk_val[0],
            )

            if ok:
                thr, cpu_du, cpu_cu, delay_total, delay_parts = msg
                env.update_network(
                    ue_id, 0, 0, 0,
                    1, env.P_ib_sk_val[0],
                    thr, cpu_du, cpu_cu, delay_total, delay_parts
                )
                accepted += 1
            else:
                reasons[str(msg)] = reasons.get(str(msg), 0) + 1

        print("accepted this step:", accepted)
        # print("fail reasons:", reasons)
        print("RB_remaining:", env.RB_remaining)
        print("PRB_remaining_per_RU:", env.PRB_remaining_per_RU)
        print("RU_power_remaining:", env.RU_power_remaining)
        print("DU_remaining:", env.DU_remaining)
        print("CU_remaining:", env.CU_remaining)
        # print_step_info(state, info)

    print("Smoke test done.")


if __name__ == "__main__":
    main()