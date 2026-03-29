from env.HOenv import HandoverEnv

import numpy as np



def print_reset_info(obs, info, env):
    print("=== RESET OK ===")
    print("obs shape:", obs.shape)
    print("obs dtype:", obs.dtype)
    print("observation_space shape:", env.observation_space.shape)
    print("action_space:", env.action_space)
    print("info:", info)
    print()


def print_step_info(step_id, action, obs, reward, terminated, truncated, info):
    print(f"=== STEP {step_id} ===")
    print("action:", action)
    print("obs shape:", obs.shape)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("info:", info)
    print()


def check_observation(env, obs):
    assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
    assert obs.shape == env.observation_space.shape, (
        f"Observation shape mismatch: got {obs.shape}, expected {env.observation_space.shape}"
    )
    assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"


def check_reward(reward):
    assert np.isfinite(reward), "Reward must be finite"


def check_info(info):
    required_keys = [
        "step",
        "candidate_count",
        "qos_violation_count",
        "mean_throughput_bps",
        "mean_latency_s",
        "prb_pool_free",
    ]
    for key in required_keys:
        assert key in info, f"Missing key in info: {key}"


def run_random_policy_test(n_steps: int = 5):
    env = HandoverEnv()

    obs, info = env.reset()
    check_observation(env, obs)
    check_info(info)
    print_reset_info(obs, info, env)

    for step_id in range(n_steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        check_observation(env, obs)
        check_reward(reward)
        check_info(info)

        print_step_info(
            step_id=step_id,
            action=action,
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if terminated or truncated:
            break

    env.close()
    print("=== RANDOM POLICY TEST PASSED ===")


def run_fixed_action_test(n_steps: int = 5):
    env = HandoverEnv()

    obs, info = env.reset()
    check_observation(env, obs)
    check_info(info)

    # always choose RU 0 for all candidate slots
    action = np.zeros(env.max_candidate_ue, dtype=np.int32)

    print("=== FIXED ACTION TEST ===")
    print("fixed action:", action)
    print()

    for step_id in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(action)

        check_observation(env, obs)
        check_reward(reward)
        check_info(info)

        print_step_info(
            step_id=step_id,
            action=action,
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if terminated or truncated:
            break

    env.close()
    print("=== FIXED ACTION TEST PASSED ===")


def run_internal_state_check():
    env = HandoverEnv()
    obs, info = env.reset()

    print("=== INTERNAL STATE CHECK ===")
    print("ue_pos shape:", env.ue_pos.shape)
    print("ue_vel shape:", env.ue_vel.shape)
    print("ue_slice shape:", env.ue_slice.shape)
    print("queue_bits shape:", env.queue_bits.shape)

    assert env.topology is not None, "Topology was not initialized"
    assert env.resource_state is not None, "Resource state was not initialized"
    assert env.last_state is not None, "Last state was not initialized"

    rs = env.resource_state
    required_rs_keys = [
        "ru_prb_allocated",
        "ru_used_prb",
        "ru_free_prb",
        "ue_allocated_prb",
        "ue_power_alloc_w",
        "prb_pool_free",
        "ue_count_per_ru",
    ]
    for key in required_rs_keys:
        assert key in rs, f"Missing resource state key: {key}"

    print("resource_state keys OK")
    print("ru_prb_allocated:", rs["ru_prb_allocated"])
    print("ru_used_prb:", rs["ru_used_prb"])
    print("ru_free_prb:", rs["ru_free_prb"])
    print("prb_pool_free:", rs["prb_pool_free"])
    print()

    env.close()
    print("=== INTERNAL STATE CHECK PASSED ===")


if __name__ == "__main__":
    run_internal_state_check()
    print()
    run_random_policy_test(n_steps=5)
    print()
    run_fixed_action_test(n_steps=5)