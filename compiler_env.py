"""MCPEnvironment base and CompilerTetrisEnv with Gymnasium-style reset/step."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_pkg = Path(__file__).resolve().parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))
import evaluator as ev  # noqa: E402
import toy_ir  # noqa: E402

Action = str


class MCPEnvironment:
    def reset(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def step(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class CompilerTetrisEnv(MCPEnvironment):
    """Toy-IR compiler phase-ordering env: reset samples curriculum; step applies passes."""

    INVALID_ACTION_PENALTY: float = -5.0
    SEMANTIC_PENALTY: float = -1000.0
    MAX_INVALID_ACTIONS: int = 3
    DEFAULT_MAX_STEPS: int = 5

    def __init__(
        self,
        curriculum_level: int = 1,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.vm = toy_ir.ToyVM()
        self.curriculum_level = max(1, min(5, int(curriculum_level)))
        self.max_steps = int(max_steps) if max_steps is not None else self.DEFAULT_MAX_STEPS
        self.rng = random.Random(seed)
        self.passes = toy_ir.PASSES
        self._valid = frozenset(self.passes.keys()) | {"STOP", "DONE", "done"}
        self._bundle: Optional[Dict[str, Any]] = None
        self._original: Optional[List[dict]] = None
        self._current: Optional[List[dict]] = None
        self._baseline_cycles: int = 0
        self._prev_cycles: int = 0
        self._steps: int = 0
        self._invalid_streak: int = 0

    def _pseudoasm(self, program: List[dict]) -> str:
        if not program:
            return "; (empty program)"
        lines: List[str] = []
        for i, instr in enumerate(program):
            op = str(instr.get("op", "NOP")).upper()
            dest = instr.get("dest")
            s1, s2 = instr.get("src1"), instr.get("src2")
            if dest is not None:
                if s2 is not None:
                    rhs = f"{op}  {s1!s}, {s2!s}"
                else:
                    rhs = f"{op}  {s1!s}"
                lines.append(f"  {i:>3}:  {dest} = {rhs}")
            else:
                lines.append(f"  {i:>3}:  {op}  {s1!s}")
        return "\n".join(lines)

    def _ctx(self) -> Tuple[Dict[str, int], Dict[int, int], List[int]]:
        assert self._bundle is not None
        return (
            dict(self._bundle["initial_vars"]),
            {int(k): int(v) for k, v in self._bundle["initial_mem"].items()},
            [int(x) for x in self._bundle["observable_addrs"]],
        )

    def _count(self, prog: List[dict]) -> int:
        iv, im, _ = self._ctx()
        return self.vm.execute_and_count_cycles(prog, iv, im)

    def _verify(self, original: List[dict], candidate: List[dict]) -> bool:
        iv, im, oa = self._ctx()
        return ev.verify_equivalence(
            original, candidate, initial_vars=iv, initial_mem=im, observable_addrs=oa
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        opts = options or {}
        if "curriculum_level" in opts:
            self.curriculum_level = max(1, min(5, int(opts["curriculum_level"])))
        self._bundle = ev.generate_curriculum_bundle(self.curriculum_level, self.rng)
        self._original = copy.deepcopy(self._bundle["instructions"])
        self._current = copy.deepcopy(self._bundle["instructions"])
        self._baseline_cycles = self._count(self._current)
        self._prev_cycles = self._baseline_cycles
        self._steps = 0
        self._invalid_streak = 0
        obs = self._pseudoasm(self._current)
        info = {
            "curriculum_level": self.curriculum_level,
            "baseline_cycles": self._baseline_cycles,
            "initial_vars": dict(self._bundle["initial_vars"]),
            "initial_mem": dict(self._bundle["initial_mem"]),
            "observable_addrs": list(self._bundle["observable_addrs"]),
        }
        return obs, info

    def step(self, action: Action) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        assert self._current is not None and self._original is not None
        a = str(action).strip()
        aup = a.upper()

        if aup in ("STOP", "DONE"):
            ok = self._verify(self._original, self._current)
            cur_c = self._count(self._current)
            if not ok:
                return (
                    self._pseudoasm(self._current),
                    self.SEMANTIC_PENALTY,
                    True,
                    False,
                    {"reason": "stop_verify_failed", "equivalent": False, "cycles": cur_c},
                )
            reward = float(self._baseline_cycles - cur_c)
            return (
                self._pseudoasm(self._current),
                reward,
                True,
                False,
                {
                    "reason": "stop",
                    "equivalent": True,
                    "baseline_cycles": self._baseline_cycles,
                    "final_cycles": cur_c,
                },
            )

        if a not in self.passes:
            self._invalid_streak += 1
            terminated = self._invalid_streak >= self.MAX_INVALID_ACTIONS
            info: Dict[str, Any] = {
                "reason": "invalid_action",
                "action": a,
                "valid_actions": sorted(self._valid),
                "consecutive_invalid": self._invalid_streak,
            }
            return (
                self._pseudoasm(self._current),
                self.INVALID_ACTION_PENALTY,
                terminated,
                False,
                info,
            )

        candidate = self.passes[a](copy.deepcopy(self._current))
        if not self._verify(self._original, candidate):
            return (
                self._pseudoasm(self._current),
                self.SEMANTIC_PENALTY,
                True,
                False,
                {"reason": "semantic_violation", "action": a},
            )

        new_c = self._count(candidate)
        cycle_delta = float(self._prev_cycles - new_c)
        self._current = candidate
        self._prev_cycles = new_c
        self._steps += 1
        self._invalid_streak = 0

        truncated = self._steps >= self.max_steps
        terminated = False
        info = {
            "action": a,
            "cycle_delta": cycle_delta,
            "cycles": new_c,
            "baseline_cycles": self._baseline_cycles,
            "steps": self._steps,
        }
        if truncated:
            info["reason"] = "max_steps"
        return self._pseudoasm(self._current), cycle_delta, terminated, truncated, info
