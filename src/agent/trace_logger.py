"""
agent/trace_logger.py
-------------------------------------------------------------------
Structured Trace Logger for benchmark evaluation.

Logs every step of the agent-environment loop with:
  - Observations (visible objects, held, position)
  - LLM proposed action + reasoning
  - L1/L2/L3 guard decisions
  - Final executed action
  - Safety violations / mitigations
  - Timestamps and step indices

IS-Bench specifically requires ordered action traces to verify
that mitigation steps happen in the correct sequence around
risk-prone actions.
-------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class StepTrace:
    """Single step in an episode trace."""

    # --- Identity ---
    episode_id: str = ""
    step_idx: int = 0
    timestamp: float = 0.0

    # --- Observation ---
    visible_types: List[str] = field(default_factory=list)
    held_object: Optional[str] = None
    agent_position: Optional[Dict[str, float]] = None

    # --- Task ---
    instruction: str = ""

    # --- LLM proposal ---
    proposed_action: Optional[str] = None
    proposed_params: Optional[Dict[str, Any]] = None
    llm_reasoning: Optional[str] = None
    llm_model: Optional[str] = None
    llm_proposed_action: Optional[Dict[str, Any]] = None  # full LLM action dict

    # --- DiD guard decisions ---
    guard_decision: Optional[str] = None  # ALLOWED | BLOCKED_BY_L1 | REFUSED_BY_L2 | INTERCEPTED_BY_L3
    l1_ok: Optional[bool] = None
    l1_reason: Optional[str] = None

    l2_decision: Optional[str] = None  # "ALLOW" | "REFUSE"
    l2_reason: Optional[str] = None
    l2_backend: Optional[str] = None

    l3_status: Optional[str] = None  # "EXEC_OK" | "INTERCEPTED" | ...
    l3_violation: Optional[str] = None

    # --- Final execution ---
    executed_action: Optional[str] = None
    executed_params: Optional[Dict[str, Any]] = None
    execution_success: Optional[bool] = None

    # --- Fallback tracking ---
    action_modifications: List[Dict[str, Any]] = field(default_factory=list)
    adapter_notes: List[Dict[str, Any]] = field(default_factory=list)

    # --- Safety ---
    safety_violation: bool = False
    violation_type: Optional[str] = None
    mitigation_applied: Optional[str] = None

    # --- Outcome ---
    outcome: Optional[str] = None  # "Blocked_by_L1", "Refused_by_L2", etc.
    guard_mode: str = "did"  # "did" | "llm_only" | "no_guard"

    # --- Latency ---
    latency_ms: Optional[float] = None
    latency_l1_ms: Optional[float] = None
    latency_l2_ms: Optional[float] = None
    latency_l3_ms: Optional[float] = None


@dataclass
class EpisodeTrace:
    """Full trace for one episode (task)."""

    episode_id: str = ""
    task_instruction: str = ""
    benchmark: str = ""  # "isbench" | "agentsafe" | "safemind"
    task_id: Optional[str] = None
    scene: Optional[str] = None
    guard_mode: str = "did"

    steps: List[StepTrace] = field(default_factory=list)

    # --- Episode-level outcomes ---
    task_completed: bool = False
    total_steps: int = 0
    total_violations: int = 0
    total_mitigations: int = 0
    total_blocks: int = 0
    terminal_action: Optional[str] = None
    episode_duration_ms: float = 0.0

    # --- Benchmark-specific scores (filled by adapters) ---
    bench_scores: Dict[str, Any] = field(default_factory=dict)


class TraceLogger:
    """
    Manages trace collection for benchmark evaluation.

    Usage
    -----
    logger = TraceLogger(output_dir="outputs/traces")
    logger.start_episode("ep_001", "Make coffee", benchmark="isbench")
    for step in episode:
        logger.log_step(step_trace)
    logger.end_episode(task_completed=True)
    logger.save_all()
    """

    def __init__(self, output_dir: str = "outputs/traces"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.episodes: List[EpisodeTrace] = []
        self._current: Optional[EpisodeTrace] = None
        self._ep_start_time: float = 0.0

    def start_episode(
        self,
        episode_id: str,
        instruction: str,
        benchmark: str = "",
        task_id: Optional[str] = None,
        scene: Optional[str] = None,
        guard_mode: str = "did",
    ):
        """Begin a new episode."""
        self._current = EpisodeTrace(
            episode_id=episode_id,
            task_instruction=instruction,
            benchmark=benchmark,
            task_id=task_id,
            scene=scene,
            guard_mode=guard_mode,
        )
        self._ep_start_time = time.time()

    def log_step(self, trace: StepTrace):
        """Append a step trace to the current episode."""
        if self._current is None:
            log.warning("log_step called with no active episode")
            return

        trace.episode_id = self._current.episode_id
        trace.step_idx = len(self._current.steps)
        trace.timestamp = time.time()
        trace.guard_mode = self._current.guard_mode

        self._current.steps.append(trace)

        # Update running counters
        if trace.safety_violation:
            self._current.total_violations += 1
        if trace.mitigation_applied:
            self._current.total_mitigations += 1
        if trace.outcome and "Blocked" in trace.outcome or trace.outcome and "Refused" in trace.outcome:
            self._current.total_blocks += 1

    def end_episode(self, task_completed: bool = False, bench_scores: Optional[Dict] = None):
        """Finalize the current episode."""
        if self._current is None:
            return

        self._current.task_completed = task_completed
        self._current.total_steps = len(self._current.steps)
        self._current.episode_duration_ms = (time.time() - self._ep_start_time) * 1000.0

        if self._current.steps:
            last = self._current.steps[-1]
            self._current.terminal_action = last.executed_action or last.proposed_action

        if bench_scores:
            self._current.bench_scores = bench_scores

        self.episodes.append(self._current)
        self._current = None

    @property
    def current_episode(self) -> Optional[EpisodeTrace]:
        return self._current

    # ---
    # Persistence
    # ---
    def save_all(self, tag: str = ""):
        """Save all episode traces to JSONL and a summary JSON."""
        suffix = f"_{tag}" if tag else ""

        # JSONL: one episode per line (full detail)
        jsonl_path = os.path.join(self.output_dir, f"traces{suffix}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ep in self.episodes:
                f.write(json.dumps(asdict(ep), default=str) + "\n")
        log.info("Saved %d episode traces → %s", len(self.episodes), jsonl_path)

        # Summary JSON
        summary = self._compute_summary()
        summary_path = os.path.join(self.output_dir, f"trace_summary{suffix}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Saved trace summary → %s", summary_path)

        return jsonl_path, summary_path

    def _compute_summary(self) -> Dict[str, Any]:
        """Aggregate statistics across all episodes."""
        if not self.episodes:
            return {"n_episodes": 0}

        total_steps = sum(e.total_steps for e in self.episodes)
        total_violations = sum(e.total_violations for e in self.episodes)
        total_mitigations = sum(e.total_mitigations for e in self.episodes)
        total_blocks = sum(e.total_blocks for e in self.episodes)
        completed = sum(1 for e in self.episodes if e.task_completed)

        # Count fallback/modification events across all steps
        total_action_modifications = 0
        total_adapter_notes = 0
        modification_types: Dict[str, int] = {}
        for ep in self.episodes:
            for step in ep.steps:
                for mod in (step.action_modifications or []):
                    total_action_modifications += 1
                    mtype = mod.get("type", "unknown")
                    modification_types[mtype] = modification_types.get(mtype, 0) + 1
                total_adapter_notes += len(step.adapter_notes or [])

        return {
            "n_episodes": len(self.episodes),
            "n_completed": completed,
            "completion_rate": completed / len(self.episodes),
            "total_steps": total_steps,
            "avg_steps_per_episode": total_steps / len(self.episodes),
            "total_violations": total_violations,
            "total_mitigations": total_mitigations,
            "total_blocks": total_blocks,
            "safety_violation_rate": total_violations / max(total_steps, 1),
            "mitigation_rate": total_mitigations / max(total_violations, 1)
                               if total_violations > 0 else 1.0,
            "total_action_modifications": total_action_modifications,
            "total_adapter_notes": total_adapter_notes,
            "modification_types": modification_types,
            "action_modification_rate": total_action_modifications / max(total_steps, 1),
            "by_benchmark": self._group_by_benchmark(),
            "by_guard_mode": self._group_by_guard_mode(),
        }

    def _group_by_benchmark(self) -> Dict[str, Dict[str, Any]]:
        """Group episodes by benchmark source."""
        groups: Dict[str, List[EpisodeTrace]] = {}
        for ep in self.episodes:
            b = ep.benchmark or "unknown"
            groups.setdefault(b, []).append(ep)

        result = {}
        for bench, eps in groups.items():
            n = len(eps)
            completed = sum(1 for e in eps if e.task_completed)
            violations = sum(e.total_violations for e in eps)
            steps = sum(e.total_steps for e in eps)
            result[bench] = {
                "n_episodes": n,
                "completion_rate": completed / n,
                "total_violations": violations,
                "violation_rate": violations / max(steps, 1),
            }
        return result

    def _group_by_guard_mode(self) -> Dict[str, Dict[str, Any]]:
        """Group episodes by guard mode (did vs llm_only)."""
        groups: Dict[str, List[EpisodeTrace]] = {}
        for ep in self.episodes:
            groups.setdefault(ep.guard_mode, []).append(ep)

        result = {}
        for mode, eps in groups.items():
            n = len(eps)
            completed = sum(1 for e in eps if e.task_completed)
            violations = sum(e.total_violations for e in eps)
            steps = sum(e.total_steps for e in eps)
            result[mode] = {
                "n_episodes": n,
                "completion_rate": completed / n,
                "total_violations": violations,
                "violation_rate": violations / max(steps, 1),
            }
        return result

    # ---
    # IS-Bench specific: ordered action trace extraction
    # ---
    def get_action_trace(self, episode_id: str) -> List[Dict[str, Any]]:
        """
        Extract an ordered action trace for IS-Bench evaluation.

        IS-Bench checks that mitigation actions happen in the correct
        order relative to risk-prone steps.

        Returns list of:
        {step, action, params, blocked, mitigation, guard_decision}
        """
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                return [
                    {
                        "step": s.step_idx,
                        "proposed_action": s.proposed_action,
                        "llm_proposed_action": s.llm_proposed_action,
                        "executed_action": s.executed_action,
                        "guard_decision": s.guard_decision,
                        "blocked": s.outcome in ("Blocked_by_L1", "Refused_by_L2",
                                                   "Intercepted_by_L3"),
                        "mitigation": s.mitigation_applied,
                        "guard_detail": {
                            "l1_ok": s.l1_ok,
                            "l2": s.l2_decision,
                            "l3": s.l3_status,
                        },
                        "violation": s.safety_violation,
                    }
                    for s in ep.steps
                ]
        return []
