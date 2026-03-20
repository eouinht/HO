from __future__ import annotations

from typing import Dict, Tuple

from models import HandoverCosts, RewardWeights, TimeStep, Topology, UEAction


class RewardEngine:
    def __init__(
        self,
        weights: RewardWeights,
        ho_costs: HandoverCosts,
        delay_threshold_ms: float = 10.0,
    ) -> None:
        self.weights = weights
        self.ho_costs = ho_costs
        self.delay_threshold_ms = delay_threshold_ms

    def compute(
        self,
        prev_snapshot: TimeStep,
        next_snapshot: TimeStep,
        actions: Dict[int, UEAction],
        topology: Topology,
    ) -> Tuple[float, Dict[str, float]]:
        total_tput = 0.0
        total_queue = 0.0
        total_delay_penalty = 0.0
        total_ho_cost = 0.0

        ue_count = max(len(next_snapshot.ue_metrics), 1)

        for ue_id, next_ue in next_snapshot.ue_metrics.items():
            prev_ue = prev_snapshot.ue_metrics.get(ue_id)
            action = actions.get(ue_id)

            total_tput += float(next_ue.tput_mbps or 0.0)
            total_queue += float(next_ue.bsr_bytes or 0.0)
            latency_ms = float(next_ue.latency_ms or 0.0)
            total_delay_penalty += max(0.0, latency_ms - self.delay_threshold_ms)

            if prev_ue is not None and action is not None:
                ho_type = topology.classify_handover(prev_ue.serving_ru, action.target_ru)
                total_ho_cost += self.ho_costs.get(ho_type)

        avg_delay_penalty = total_delay_penalty / ue_count
        reward = (
            self.weights.throughput * total_tput
            - self.weights.delay * avg_delay_penalty
            - self.weights.queue * total_queue
            - self.weights.handover * total_ho_cost
        )

        info = {
            "total_tput_mbps": total_tput,
            "total_queue_bytes": total_queue,
            "avg_delay_penalty_ms": avg_delay_penalty,
            "total_handover_cost": total_ho_cost,
        }
        return reward, info