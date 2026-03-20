from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from models import EnvConfig, HandoverCosts, RewardWeights, TimeStep, Topology, TraceBundle, UEAction
from reward_engine import RewardEngine
from state_builder import StateBuilder


class TraceDrivenHandoverEnv:
    def __init__(
        self,
        trace_bundle: TraceBundle,
        env_config: Optional[EnvConfig] = None,
        reward_weights: Optional[RewardWeights] = None,
        ho_costs: Optional[HandoverCosts] = None,
    ) -> None:
        self.trace_bundle = trace_bundle
        self.env_config = env_config or EnvConfig()
        self.state_builder = StateBuilder()
        self.reward_engine = RewardEngine(
            weights=reward_weights or RewardWeights(),
            ho_costs=ho_costs or HandoverCosts(),
            delay_threshold_ms=self.env_config.delay_threshold_ms,
        )
        self._idx = 0

    @property
    def topology(self) -> Topology:
        return self.trace_bundle.topology

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.trace_bundle.steps:
            raise ValueError("Trace bundle has no time-step snapshots.")

        self._idx = 0
        snapshot = self.trace_bundle.steps[self._idx]
        state = self.state_builder.build(snapshot, self.topology)
        info = {"t": snapshot.t, "num_ues": len(snapshot.ue_metrics)}
        return state, info

    def step(self, actions: Dict[int, UEAction]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._idx >= len(self.trace_bundle.steps) - 1:
            current = self.trace_bundle.steps[self._idx]
            state = self.state_builder.build(current, self.topology)
            return state, 0.0, True, False, {"reason": "end_of_trace"}

        prev_snapshot = self.trace_bundle.steps[self._idx]
        self._validate_actions(prev_snapshot, actions)

        self._idx += 1
        next_snapshot = self.trace_bundle.steps[self._idx]

        reward, reward_info = self.reward_engine.compute(
            prev_snapshot=prev_snapshot,
            next_snapshot=next_snapshot,
            actions=actions,
            topology=self.topology,
        )

        state = self.state_builder.build(next_snapshot, self.topology)
        terminated = self._idx >= len(self.trace_bundle.steps) - 1
        truncated = False
        info = {
            "t": next_snapshot.t,
            "reward_info": reward_info,
            "handover_types": self._classify_actions(prev_snapshot, actions),
        }
        return state, reward, terminated, truncated, info

    def _validate_actions(self, snapshot: TimeStep, actions: Dict[int, UEAction]) -> None:
        for ue_id in snapshot.ue_metrics:
            if ue_id not in actions:
                raise ValueError(f"Missing action for ue_id={ue_id}")
            action = actions[ue_id]
            if action.target_ru not in self.topology.rus:
                raise ValueError(f"Invalid target_ru={action.target_ru} for ue_id={ue_id}")

    def _classify_actions(self, snapshot: TimeStep, actions: Dict[int, UEAction]) -> Dict[int, str]:
        result: Dict[int, str] = {}
        for ue_id, ue in snapshot.ue_metrics.items():
            action = actions.get(ue_id)
            if action is None:
                continue
            ho_type = self.topology.classify_handover(ue.serving_ru, action.target_ru)
            result[ue_id] = ho_type.name
        return result