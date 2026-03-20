from pathlib import Path

from env import TraceDrivenHandoverEnv
from models import UEAction
from parser import NS3TraceParser


def main() -> None:
    parser = NS3TraceParser()
    trace_path = Path(__file__).with_name("nr_stream.jsonl")
    trace = parser.parse_file(trace_path)

    env = TraceDrivenHandoverEnv(trace)
    state, info = env.reset()
    print("Reset info:", info)
    print("Initial time step:", state["t"])

    done = False
    step_count = 0
    while not done:
        actions = {}
        for ue_id, ue_state in state["ues"].items():
            serving_ru = ue_state["serving_ru"]
            actions[ue_id] = UEAction(
                target_ru=serving_ru,
                prb_alloc=0.0,
                ptx_alloc=0.0,
                du_alloc=0.0,
                cu_alloc=0.0,
            )

        state, reward, terminated, truncated, step_info = env.step(actions)
        print(f"t={state['t']}, reward={reward:.3f}, handover_types={step_info['handover_types']}")
        step_count += 1
        done = terminated or truncated

    print("Finished after", step_count, "steps")


if __name__ == "__main__":
    main()