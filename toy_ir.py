"""Toy-IR virtual machine, per-op cycle costs, and optimization passes."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

# --- cycle costs (ADD cheap, MUL/DIV expensive) --------------------------------

CYCLE_COST: Dict[str, int] = {
    "CONST": 1,
    "LOAD": 2,
    "STORE": 2,
    "ADD": 1,
    "SUB": 1,
    "MUL": 3,
    "DIV": 4,
}


def _op_u(ins: dict) -> str:
    return str(ins.get("op", "")).upper()


class ToyVM:
    """Linear Toy-IR executor with deterministic memory + cycle accounting."""

    def run(
        self,
        program: List[dict],
        initial_vars: Dict[str, int],
        initial_mem: Dict[int, int],
    ) -> Tuple[Dict[int, int], Dict[str, int], int]:
        regs = {str(k): int(v) for k, v in initial_vars.items()}
        mem = {int(k): int(v) for k, v in initial_mem.items()}
        cycles = 0

        def val(x: Any) -> int:
            if isinstance(x, str):
                return int(regs[x])
            return int(x)

        for ins in program:
            op = _op_u(ins)
            cycles += int(CYCLE_COST.get(op, 1))
            if op == "CONST":
                regs[str(ins["dest"])] = int(ins["src1"])
            elif op == "LOAD":
                idx = val(ins["src1"])
                regs[str(ins["dest"])] = int(mem[idx])
            elif op == "ADD":
                regs[str(ins["dest"])] = val(ins["src1"]) + val(ins["src2"])
            elif op == "SUB":
                regs[str(ins["dest"])] = val(ins["src1"]) - val(ins["src2"])
            elif op == "MUL":
                regs[str(ins["dest"])] = val(ins["src1"]) * val(ins["src2"])
            elif op == "DIV":
                a, b = val(ins["src1"]), val(ins["src2"])
                if b == 0:
                    regs[str(ins["dest"])] = 0
                else:
                    regs[str(ins["dest"])] = int(a // b)
            elif op == "STORE":
                addr = val(ins["dest"])
                mem[int(addr)] = val(ins["src1"])
        return mem, regs, cycles

    def execute_and_count_cycles(
        self,
        program: List[dict],
        initial_vars: Dict[str, int],
        initial_mem: Dict[int, int],
    ) -> int:
        _, _, c = self.run(program, initial_vars, initial_mem)
        return max(0, int(c))


# --- forward / reverse passes -------------------------------------------------


def constant_folding(program: List[dict]) -> List[dict]:
    p = copy.deepcopy(program)
    for i, ins in enumerate(p):
        op = _op_u(ins)
        if op not in ("ADD", "SUB", "MUL", "DIV"):
            continue
        s1, s2 = ins.get("src1"), ins.get("src2")
        if isinstance(s1, int) and isinstance(s2, int):
            if op == "ADD":
                v = s1 + s2
            elif op == "SUB":
                v = s1 - s2
            elif op == "MUL":
                v = s1 * s2
            else:
                v = 0 if s2 == 0 else int(s1 // s2)
            p[i] = {"op": "CONST", "dest": ins["dest"], "src1": v, "src2": None}
            return p
    return p


def _uses_in_ins(ins: dict) -> List[str]:
    out: List[str] = []
    for k in ("src1", "src2"):
        v = ins.get(k)
        if isinstance(v, str):
            out.append(v)
    return out


def dead_code_elimination(program: List[dict]) -> List[dict]:
    """Remove one CONST whose destination is never used (safe for curriculum IR)."""
    p = copy.deepcopy(program)
    all_uses: set = set()
    for ins in p:
        all_uses.update(_uses_in_ins(ins))
        if _op_u(ins) == "STORE":
            d = ins.get("dest")
            if isinstance(d, str):
                all_uses.add(d)
    for i, ins in enumerate(p):
        if _op_u(ins) != "CONST":
            continue
        dest = str(ins.get("dest", ""))
        if dest and dest not in all_uses:
            del p[i]
            return p
    return p


def peephole_optimization(program: List[dict]) -> List[dict]:
    return copy.deepcopy(program)


def expand_constant(program: List[dict]) -> List[dict]:
    p = copy.deepcopy(program)
    for i, ins in enumerate(p):
        if _op_u(ins) != "CONST":
            continue
        v = ins.get("src1")
        if not isinstance(v, int) or v < 2:
            continue
        d = str(ins.get("dest", "t0"))
        h, rest = v // 2, v - (v // 2)
        repl = [
            {"op": "CONST", "dest": f"{d}_a", "src1": h, "src2": None},
            {"op": "CONST", "dest": f"{d}_b", "src1": rest, "src2": None},
            {"op": "ADD", "dest": d, "src1": f"{d}_a", "src2": f"{d}_b"},
        ]
        p[i : i + 1] = repl
        return p
    return p


def duplicate_computation(program: List[dict]) -> List[dict]:
    p = copy.deepcopy(program)
    for i, ins in enumerate(p):
        ou = _op_u(ins)
        if ou not in ("ADD", "SUB", "MUL", "DIV"):
            continue
        dest = str(ins.get("dest", "t0"))
        dup = copy.deepcopy(ins)
        dup["dest"] = f"{dest}_d"
        p.insert(i + 1, dup)
        return p
    return p


PASSES: Dict[str, Callable[[List[dict]], List[dict]]] = {
    "constant_folding": constant_folding,
    "dead_code_elimination": dead_code_elimination,
    "peephole_optimization": peephole_optimization,
    "expand_constant": expand_constant,
    "duplicate_computation": duplicate_computation,
}
