"""Small case interface; all mutable fixture state lives in checkpointed world."""
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class Case:
    id: str
    title: str
    task: str
    initial: dict
    # Called after an agent turn. Return the next complete world (copy, never mutate input).
    transition: Callable[[dict, dict, bool], dict]
    # Inspect final world and accumulated turns; do not use fixed flag to assign a verdict.
    observe: Callable[[dict, list[dict]], dict]
    # Tools close over a per-turn copy of world; runner checkpoints that copy at the turn boundary.
    tools: Callable[[dict, bool], list[Any]] = lambda world, fixed: []
    prompt: Callable[[dict, bool], str] = lambda world, fixed: ""
    model_options: Callable[[dict, bool], dict] = lambda world, fixed: {}
    system_prompt: str = "Complete the user's local task using available tools. Be concise. Report uncertainty honestly."
    fidelity: str = "source_inspired_mechanism_model"
    limitation: str = "Ported local fixture; not an execution of the historical product/version."
    intervention: str = "See case implementation."
    max_cycles: int = 3
    max_model_calls: int = 4
    max_tool_calls: int = 6
    # Optional graph options, e.g. interrupt_on. Runner automatically rejects test approvals.
    graph_options: dict = field(default_factory=dict)


def verdict(observed: bool, **measurements):
    return {"symptom_observed": bool(observed), "measurements": measurements}
