"""Strict observable-memory verifier and Toy-IR curriculum (levels 1–5)."""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_pkg = Path(__file__).resolve().parent
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))
import toy_ir  # noqa: E402


def observable_snapshot(
    program: List[dict],
    initial_vars: Dict[str, int],
    initial_mem: Dict[int, int],
    observable_addrs: List[int],
) -> Tuple[Dict[int, int], int]:
    mem, _, cyc = toy_ir.ToyVM().run(program, initial_vars, initial_mem)
    snap = {int(a): int(mem.get(int(a), 0)) for a in observable_addrs}
    return snap, int(cyc)


def verify_equivalence(
    original: List[dict],
    candidate: List[dict],
    *,
    initial_vars: Dict[str, int],
    initial_mem: Dict[int, int],
    observable_addrs: List[int],
) -> bool:
    m0, _ = observable_snapshot(original, initial_vars, initial_mem, observable_addrs)
    m1, _ = observable_snapshot(candidate, initial_vars, initial_mem, observable_addrs)
    return m0 == m1


def generate_curriculum_bundle(level: int, rng: random.Random | None = None) -> dict:
    lv = max(1, min(int(level), len(ALL_LEVELS)))
    if rng is None:
        return copy.deepcopy(ALL_LEVELS[lv - 1]())
    st = random.getstate()
    random.setstate(rng.getstate())
    try:
        out = ALL_LEVELS[lv - 1]()
    finally:
        rng.setstate(random.getstate())
        random.setstate(st)
    return copy.deepcopy(out)


def generate_level_1():
    """
    Generate a Level 1 Toy-IR program.

    Characteristics:
    - 4-6 instructions
    - 2-3 CONST ops with literal values
    - 1-2 arithmetic ops on those constants (foldable by CF)
    - 0-1 dead variables (killable by DCE)
    - Exactly 1 STORE at the end so the program has an observable output

    Returns:
        dict with keys: initial_vars, initial_mem, instructions, observable_addrs
    """
    instructions = []
    var_counter = 0

    def new_var():
        nonlocal var_counter
        name = f"v{var_counter}"
        var_counter += 1
        return name

    # Step 1: Generate 2-3 constant assignments
    num_consts = random.randint(2, 3)
    const_vars = []
    for _ in range(num_consts):
        var = new_var()
        value = random.randint(1, 10)
        instructions.append({"op": "CONST", "dest": var, "src1": value, "src2": None})
        const_vars.append(var)

    # Step 2: Generate 1-2 arithmetic ops using those constants
    num_arith = random.randint(1, 2)
    last_result = None
    for _ in range(num_arith):
        op = random.choice(["ADD", "MUL"])
        src1 = random.choice(const_vars)
        src2 = random.choice(const_vars)
        dest = new_var()
        instructions.append({"op": op, "dest": dest, "src1": src1, "src2": src2})
        last_result = dest
        const_vars.append(dest)

    # Step 3: Optionally add 1 dead variable (50% chance)
    if random.random() < 0.5:
        dead_var = new_var()
        dead_value = random.randint(1, 10)
        instructions.append({"op": "CONST", "dest": dead_var, "src1": dead_value, "src2": None})
        # Note: dead_var is intentionally never used — DCE should catch it.

    # Step 4: Add a STORE at the end so the program has observable output
    initial_vars = {"addr0": 0}
    instructions.append({"op": "STORE", "dest": "addr0", "src1": last_result, "src2": None})

    initial_mem = {}

    return {
        "initial_vars":     initial_vars,
        "initial_mem":      initial_mem,
        "instructions":     instructions,
        "observable_addrs": [0],   # Aarush's verifier compares only these mem entries
    }

def generate_level_2():
    """
    Generate a Level 2 Toy-IR program.

    Characteristics:
    - 8-12 instructions
    - 3-5 CONSTs (some literal, some used in arithmetic)
    - 3-5 arithmetic ops (ADD, SUB, MUL, DIV) with chaining
    - 1-3 dead variables (DCE opportunities)
    - 1 LOAD from initial memory whose result is GUARANTEED to flow into
      the final STORE (forces agent to reason around unfoldable runtime values)
    - 1 STORE at the end (observable output)
    - DIV always uses literal non-zero divisor (Design Assumption #4)

    Returns:
        dict with keys: initial_vars, initial_mem, instructions, observable_addrs
    """
    instructions = []
    var_counter = 0

    def new_var():
        nonlocal var_counter
        name = f"v{var_counter}"
        var_counter += 1
        return name

    available_vars = []

    # Step 1: 3-5 CONSTs
    num_consts = random.randint(3, 5)
    for _ in range(num_consts):
        var = new_var()
        value = random.randint(1, 10)
        instructions.append({"op": "CONST", "dest": var, "src1": value, "src2": None})
        available_vars.append(var)

    # Step 2: 1 LOAD from initial memory
    load_addr_var = "addr_in"
    loaded_var    = new_var()
    instructions.append({
        "op": "LOAD", "dest": loaded_var, "src1": load_addr_var, "src2": None,
    })
    available_vars.append(loaded_var)

    # Step 3: 3-5 arithmetic ops, chained
    num_arith = random.randint(3, 5)
    last_result = None
    for _ in range(num_arith):
        op = random.choice(["ADD", "SUB", "MUL", "DIV"])
        use_literal_src2 = random.random() < 0.3

        src1 = random.choice(available_vars)
        if use_literal_src2:
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4])
            else:
                src2 = random.choice([0, 1, 2, 3])
        else:
            src2 = random.choice(available_vars)
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4])

        dest = new_var()
        instructions.append({"op": op, "dest": dest, "src1": src1, "src2": src2})
        available_vars.append(dest)
        last_result = dest

    # Step 4: 1-3 extra dead CONSTs (DCE targets)
    num_dead = random.randint(1, 3)
    for _ in range(num_dead):
        dead_var   = new_var()
        dead_value = random.randint(1, 20)
        insert_pos = random.randint(0, len(instructions))
        instructions.insert(insert_pos, {
            "op": "CONST", "dest": dead_var, "src1": dead_value, "src2": None,
        })

    # === Step 5 (NEW): force LOAD-into-chain dependency ===
    # Inject a final ADD that combines last_result with the loaded variable.
    # This guarantees the LOAD result flows into the STORE — DCE can no longer
    # eliminate it, and CF cannot collapse the entire program into a CONST.
    final_dest = new_var()
    instructions.append({
        "op": "ADD",
        "dest": final_dest,
        "src1": last_result,
        "src2": loaded_var,
    })
    last_result = final_dest

    # Step 6: STORE the final result
    initial_vars = {
        "addr0":   0,
        "addr_in": 1,
    }
    initial_mem = {1: random.randint(1, 20)}

    instructions.append({
        "op": "STORE", "dest": "addr0", "src1": last_result, "src2": None,
    })

    return {
        "initial_vars":     initial_vars,
        "initial_mem":      initial_mem,
        "instructions":     instructions,
        "observable_addrs": [0],
    }

def generate_level_3():
    """
    Generate a Level 3 Toy-IR program.

    Characteristics:
    - 15-20 instructions
    - 5-7 CONSTs
    - 2 LOADs (both forced to flow into final STORE)
    - 6-9 arithmetic ops with deeper chaining
    - 2-4 dead variables sprinkled throughout
    - 1 main STORE (optionally a secondary STORE)
    - DIV always uses literal non-zero divisor (Design Assumption #4)

    Returns:
        dict with keys: initial_vars, initial_mem, instructions, observable_addrs
    """
    instructions = []
    var_counter = 0

    def new_var():
        nonlocal var_counter
        name = f"v{var_counter}"
        var_counter += 1
        return name

    available_vars = []

    # Step 1: 5-7 CONSTs
    num_consts = random.randint(5, 7)
    for _ in range(num_consts):
        var = new_var()
        value = random.randint(1, 15)
        instructions.append({"op": "CONST", "dest": var, "src1": value, "src2": None})
        available_vars.append(var)

    # Step 2: 2 LOADs from initial memory
    loaded_vars = []
    for i in range(2):
        addr_name  = f"addr_in{i}"
        loaded_var = new_var()
        instructions.append({
            "op": "LOAD", "dest": loaded_var, "src1": addr_name, "src2": None,
        })
        available_vars.append(loaded_var)
        loaded_vars.append(loaded_var)

    # Step 3: 6-9 arithmetic ops, chained
    num_arith = random.randint(6, 9)
    last_result = None
    for _ in range(num_arith):
        op = random.choice(["ADD", "SUB", "MUL", "DIV"])
        use_literal_src2 = random.random() < 0.3

        src1 = random.choice(available_vars)
        if use_literal_src2:
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4, 5])
            else:
                src2 = random.choice([0, 1, 2, 3])
        else:
            src2 = random.choice(available_vars)
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4, 5])

        dest = new_var()
        instructions.append({"op": op, "dest": dest, "src1": src1, "src2": src2})
        available_vars.append(dest)
        last_result = dest

    # Step 4: 2-4 dead CONSTs scattered throughout
    num_dead = random.randint(2, 4)
    for _ in range(num_dead):
        dead_var   = new_var()
        dead_value = random.randint(1, 25)
        insert_pos = random.randint(0, len(instructions))
        instructions.insert(insert_pos, {
            "op": "CONST", "dest": dead_var, "src1": dead_value, "src2": None,
        })

    # Step 5: Force BOTH LOADed variables into the final result chain.
    # Combine last_result with both loaded values via two ADDs.
    after_load1 = new_var()
    instructions.append({
        "op": "ADD",
        "dest": after_load1,
        "src1": last_result,
        "src2": loaded_vars[0],
    })

    after_load2 = new_var()
    instructions.append({
        "op": "ADD",
        "dest": after_load2,
        "src1": after_load1,
        "src2": loaded_vars[1],
    })
    last_result = after_load2

    # Step 6: Main STORE
    instructions.append({
        "op": "STORE", "dest": "addr0", "src1": last_result, "src2": None,
    })

    # Step 7 (optional, 40% chance): Secondary STORE to a different address.
    # Stores an intermediate constant — gives the agent another observable to preserve.
    has_secondary_store = random.random() < 0.4
    observable_addrs = [0]
    if has_secondary_store:
        # Pick one of the early-defined CONSTs as the secondary value
        early_const_vars = [
            instr["dest"] for instr in instructions[:num_consts]
            if instr["op"] == "CONST"
        ]
        if early_const_vars:
            secondary_src = random.choice(early_const_vars)
            instructions.append({
                "op": "STORE", "dest": "addr1", "src1": secondary_src, "src2": None,
            })
            observable_addrs.append(1)

    # Step 8: Initial state
    initial_vars = {
        "addr0":    0,
        "addr1":    1,
        "addr_in0": 2,
        "addr_in1": 3,
    }
    initial_mem = {
        2: random.randint(1, 25),
        3: random.randint(1, 25),
    }

    return {
        "initial_vars":     initial_vars,
        "initial_mem":      initial_mem,
        "instructions":     instructions,
        "observable_addrs": observable_addrs,
    }

def generate_level_4():
    """
    Generate a Level 4 Toy-IR program with fan-out topology.

    Distinct from Levels 1-3: 2-3 'hub' variables fan out to drive
    most arithmetic operations. A single optimization at a hub cascades
    through many dependents, giving the agent measurable leverage from
    correctly identifying high-value optimization targets.

    Characteristics:
    - 20-28 instructions
    - 5-7 CONSTs (some become hubs)
    - 2 LOADs (one usually a hub)
    - 10-14 arithmetic ops, ~70% using a hub variable as src1
    - 3-5 dead variables
    - 1-2 STOREs (both LOADs flow into final result)

    Returns:
        dict with keys: initial_vars, initial_mem, instructions, observable_addrs
    """
    instructions = []
    var_counter = 0

    def new_var():
        nonlocal var_counter
        name = f"v{var_counter}"
        var_counter += 1
        return name

    available_vars = []

    # Step 1: 5-7 CONSTs
    num_consts = random.randint(5, 7)
    const_vars = []
    for _ in range(num_consts):
        var = new_var()
        value = random.randint(1, 15)
        instructions.append({"op": "CONST", "dest": var, "src1": value, "src2": None})
        available_vars.append(var)
        const_vars.append(var)

    # Step 2: 2 LOADs
    loaded_vars = []
    for i in range(2):
        addr_name  = f"addr_in{i}"
        loaded_var = new_var()
        instructions.append({
            "op": "LOAD", "dest": loaded_var, "src1": addr_name, "src2": None,
        })
        available_vars.append(loaded_var)
        loaded_vars.append(loaded_var)

    # Step 3: Designate 2-3 hub variables.
    # Hubs come from a mix of constants and loads to ensure CF can exploit
    # constant hubs while non-constant (LOAD) hubs force the agent to reason
    # about partially-resolvable structure.
    num_hubs = random.randint(2, 3)
    candidate_hubs = const_vars[:3] + loaded_vars  # bias toward early-defined vars
    hub_vars = random.sample(candidate_hubs, min(num_hubs, len(candidate_hubs)))

    # Step 4: 10-14 arithmetic ops, ~70% using a hub as src1
    num_arith = random.randint(10, 14)
    last_result = None
    for _ in range(num_arith):
        op = random.choice(["ADD", "SUB", "MUL", "DIV"])
        use_literal_src2 = random.random() < 0.3

        # 70% chance src1 is a hub (drives the fan-out structure)
        if random.random() < 0.70:
            src1 = random.choice(hub_vars)
        else:
            src1 = random.choice(available_vars)

        if use_literal_src2:
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4, 5])
            else:
                src2 = random.choice([0, 1, 2, 3])
        else:
            src2 = random.choice(available_vars)
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4, 5])

        dest = new_var()
        instructions.append({"op": op, "dest": dest, "src1": src1, "src2": src2})
        available_vars.append(dest)
        last_result = dest

    # Step 5: 3-5 dead CONSTs scattered throughout
    num_dead = random.randint(3, 5)
    for _ in range(num_dead):
        dead_var   = new_var()
        dead_value = random.randint(1, 30)
        insert_pos = random.randint(0, len(instructions))
        instructions.insert(insert_pos, {
            "op": "CONST", "dest": dead_var, "src1": dead_value, "src2": None,
        })

    # Step 6: Force both LOADed values into the final chain
    after_load1 = new_var()
    instructions.append({
        "op": "ADD", "dest": after_load1,
        "src1": last_result, "src2": loaded_vars[0],
    })
    after_load2 = new_var()
    instructions.append({
        "op": "ADD", "dest": after_load2,
        "src1": after_load1, "src2": loaded_vars[1],
    })
    last_result = after_load2

    # Step 7: Main STORE
    instructions.append({
        "op": "STORE", "dest": "addr0", "src1": last_result, "src2": None,
    })

    # Step 8 (40% chance): secondary STORE
    has_secondary_store = random.random() < 0.4
    observable_addrs = [0]
    if has_secondary_store:
        early_const_vars = [
            instr["dest"] for instr in instructions[:num_consts]
            if instr["op"] == "CONST"
        ]
        if early_const_vars:
            secondary_src = random.choice(early_const_vars)
            instructions.append({
                "op": "STORE", "dest": "addr1", "src1": secondary_src, "src2": None,
            })
            observable_addrs.append(1)

    # Step 9: Initial state
    initial_vars = {
        "addr0":    0,
        "addr1":    1,
        "addr_in0": 2,
        "addr_in1": 3,
    }
    initial_mem = {
        2: random.randint(1, 30),
        3: random.randint(1, 30),
    }

    return {
        "initial_vars":     initial_vars,
        "initial_mem":      initial_mem,
        "instructions":     instructions,
        "observable_addrs": observable_addrs,
    }

def generate_level_5():
    """
    Generate a Level 5 Toy-IR program — maximum adversarial density.

    Level 5 is the hardest curriculum tier. Programs are long (30-40
    instructions), peppered with dead code, heavy on expensive ops
    (MUL/DIV), and feature multiple LOADs and STOREs. The 70% literal-bait
    rate ensures peephole has many opportunities, and the deep arithmetic
    chains create real differentiation between optimization strategies.

    Distinct from Level 4: longer, denser, more peephole bait, more
    expensive operations, multiple observable outputs. The absolute cycle
    savings are huge, magnifying any sub-optimal pass orderings.

    Returns:
        dict with keys: initial_vars, initial_mem, instructions, observable_addrs
    """
    instructions = []
    var_counter = 0

    def new_var():
        nonlocal var_counter
        name = f"v{var_counter}"
        var_counter += 1
        return name

    available_vars = []

    # Step 1: 7-9 CONSTs
    num_consts = random.randint(7, 9)
    const_vars = []
    for _ in range(num_consts):
        var = new_var()
        value = random.randint(1, 20)
        instructions.append({"op": "CONST", "dest": var, "src1": value, "src2": None})
        available_vars.append(var)
        const_vars.append(var)

    # Step 2: 3 LOADs
    loaded_vars = []
    for i in range(3):
        addr_name  = f"addr_in{i}"
        loaded_var = new_var()
        instructions.append({
            "op": "LOAD", "dest": loaded_var, "src1": addr_name, "src2": None,
        })
        available_vars.append(loaded_var)
        loaded_vars.append(loaded_var)

    # Step 3: Designate 3-4 hub variables
    num_hubs = random.randint(3, 4)
    candidate_hubs = const_vars[:5] + loaded_vars
    hub_vars = random.sample(candidate_hubs, min(num_hubs, len(candidate_hubs)))

    # Step 4: 15-22 arithmetic ops, heavy on expensive ops + peephole bait
    num_arith = random.randint(15, 22)
    last_result = None
    for _ in range(num_arith):
        # Bias toward expensive ops (MUL/DIV) for higher cycle savings on optimization
        op = random.choices(
            ["ADD", "SUB", "MUL", "DIV"],
            weights=[2, 2, 3, 2],   # MUL slightly preferred
        )[0]

        # 70% chance of literal src2 → peephole bait
        use_literal_src2 = random.random() < 0.70

        # 60% chance src1 is a hub
        if random.random() < 0.60:
            src1 = random.choice(hub_vars)
        else:
            src1 = random.choice(available_vars)

        if use_literal_src2:
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4, 5])    # peephole bait: 1
            else:
                src2 = random.choice([0, 1, 2, 3, 4])    # peephole bait: 0, 1, 2
        else:
            src2 = random.choice(available_vars)
            if op == "DIV":
                src2 = random.choice([1, 2, 3, 4, 5])

        dest = new_var()
        instructions.append({"op": op, "dest": dest, "src1": src1, "src2": src2})
        available_vars.append(dest)
        last_result = dest

    # Step 5: 5-8 dead CONSTs
    num_dead = random.randint(5, 8)
    for _ in range(num_dead):
        dead_var   = new_var()
        dead_value = random.randint(1, 50)
        insert_pos = random.randint(0, len(instructions))
        instructions.insert(insert_pos, {
            "op": "CONST", "dest": dead_var, "src1": dead_value, "src2": None,
        })

    # Step 6: Force all 3 LOADed values into the final chain
    after_load1 = new_var()
    instructions.append({
        "op": "ADD", "dest": after_load1,
        "src1": last_result, "src2": loaded_vars[0],
    })
    after_load2 = new_var()
    instructions.append({
        "op": "ADD", "dest": after_load2,
        "src1": after_load1, "src2": loaded_vars[1],
    })
    after_load3 = new_var()
    instructions.append({
        "op": "ADD", "dest": after_load3,
        "src1": after_load2, "src2": loaded_vars[2],
    })
    last_result = after_load3

    # Step 7: Multiple STOREs (always 2-3)
    num_stores = random.randint(2, 3)

    # Main STORE
    instructions.append({
        "op": "STORE", "dest": "addr0", "src1": last_result, "src2": None,
    })
    observable_addrs = [0]

    # Secondary STOREs use early CONSTs (gives DCE more variety to handle)
    early_const_vars = [
        instr["dest"] for instr in instructions[:num_consts]
        if instr["op"] == "CONST"
    ]
    for i in range(1, num_stores):
        if early_const_vars:
            secondary_src = random.choice(early_const_vars)
            addr_idx = i  # addr1, addr2
            instructions.append({
                "op": "STORE", "dest": f"addr{addr_idx}", "src1": secondary_src, "src2": None,
            })
            observable_addrs.append(addr_idx)

    # Step 8: Initial state
    initial_vars = {
        "addr0":    0,
        "addr1":    1,
        "addr2":    2,
        "addr_in0": 3,
        "addr_in1": 4,
        "addr_in2": 5,
    }
    initial_mem = {
        3: random.randint(1, 50),
        4: random.randint(1, 50),
        5: random.randint(1, 50),
    }

    return {
        "initial_vars":     initial_vars,
        "initial_mem":      initial_mem,
        "instructions":     instructions,
        "observable_addrs": observable_addrs,
    }

ALL_LEVELS = [generate_level_1, generate_level_2, generate_level_3, generate_level_4, generate_level_5]
