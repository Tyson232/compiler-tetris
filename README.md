# Compiler Tetris

An RL agent that learns to optimize compiler IR by deciding *which* optimization passes to run and *in what order* — and, crucially, when to deliberately make the code **worse** before making it better.

This repo contains the compiler environment, the verifier, and a small Gradio demo. The training pipeline (GRPO + Unsloth) is documented below and lives in the linked training repo.

---

## The one-paragraph version

Compilers spend a lot of effort on "phase ordering" — the order in which you run optimization passes. The order matters a lot, and for decades it's been driven by hand-written heuristics. We treat phase ordering as a reinforcement learning problem: the agent looks at a program's intermediate representation (IR), picks a pass to apply, sees how the cycle count changes, and keeps going until it decides to stop. The reward is real cycles saved, and any transformation that changes what the program actually computes is punished hard. The twist that makes this more than a toy: we gave the agent two *reverse* passes that intentionally raise the cycle count. Counter-intuitive, but it's what lets the policy climb out of local minima and find optimizations a forward-only pipeline mathematically cannot reach.

## Why IR optimization, and why RL

If you look at where compiler research has gone, almost every stage of the pipeline has been heavily tuned. The mid-level IR optimization step — where the compiler simplifies a structured, mid-level form of your code before lowering it to machine instructions — is one of the places where the "right" sequence of transformations is still mostly heuristic. That's exactly the shape of problem RL is good at: a large discrete action space, delayed payoff, and a clean, verifiable reward signal. You either preserved the program's behavior or you didn't, and you either saved cycles or you didn't. No human labels required.

## The reverse pass (the part we actually care about)

A pure forward optimizer can only ever *reduce* the cycle count. That sounds fine until you realize it traps the policy in a local minimum — it grabs the first good-looking solution and gets stuck. We took the idea from a paper called *Beyond Phase Ordering*: deliberately push the IR toward a *worse* state first, then optimize back up from there. Optimizing the original code directly might get you partway. Wrecking it first and then re-optimizing gets you to the real global optimum.

The analogy that made it click for us: you're standing on a hill and want the tallest peak. There's a small peak right in front of you, so you climb it and feel good. But if you'd first walked *down* into the valley, you'd have seen the whole landscape — and the actual highest peak across the way. The valley isn't a setback, it's the only vantage point from which the global maximum is visible.

We implemented two reverse passes:

- **`expand_constant`** — splits a constant into two halves plus an `ADD` (e.g. `c = 10` becomes `a = 5; b = 5; c = a + b`). Costs one extra cycle on purpose.
- **`duplicate_computation`** — inserts a redundant copy of an arithmetic op.

In isolation these are bad moves. That's the point. They break apart rigid structure so the forward passes can re-fold the math in ways that weren't reachable before.

## How the environment works

The IR is linear three-address code. Every instruction is a plain dict:

```python
{"op": "MUL", "dest": "v3", "src1": "v1", "src2": "v2"}
```

A small VM (`ToyVM` in `toy_ir.py`) executes a program against an initial register/memory state and counts cycles. Costs are per-op, so the agent can't cheat by just deleting lines — different operations are worth different amounts:

| Op | Cycles |
|----|--------|
| CONST, ADD, SUB | 1 |
| LOAD, STORE | 2 |
| MUL | 3 |
| DIV | 4 |

`STORE`s are the program's observable output. The verifier only compares the final contents of a fixed set of memory addresses (`observable_addrs`), so the agent is free to rewrite everything else however it likes — as long as the observable result is identical.

The whole thing is wrapped in a Gym-style API (`CompilerTetrisEnv` in `compiler_env.py`):

```python
env = CompilerTetrisEnv(curriculum_level=3, max_steps=5, seed=0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step("constant_folding")
...
obs, reward, terminated, truncated, info = env.step("STOP")
```

`reset()` samples a program from the curriculum and returns it as readable pseudo-assembly (the raw dict form is too noisy for an LLM to reason over, so we render it into something closer to what a pretrained model already understands). Each `step(action)` applies one pass and returns the cycle delta as a shaped reward. `STOP` ends the episode and pays out the total cycles saved versus the baseline.

### Rewards and guardrails

- **Per-step reward** = cycles saved by that pass (can be negative — that's the reverse passes earning their keep).
- **Final reward on STOP** = `baseline_cycles − final_cycles`, but only if the program is still equivalent.
- **Semantic violation** (observable memory changed) → **−1000** and the episode ends. Correctness is never optional.
- **Invalid action** → −5; three in a row ends the episode.

## The passes

Defined in `toy_ir.py`:

**Forward (reduce cycles)**
- `constant_folding` — evaluates ops on literal operands at compile time.
- `dead_code_elimination` — traces uses backward from `STORE`s and removes computations that never reach an output.
- `peephole_optimization` — slot for local algebraic rewrites.

**Reverse (raise cycles to escape local minima)**
- `expand_constant`
- `duplicate_computation`

Each forward pass applies a single transformation per call and returns, so the agent has to *chain* them — which is exactly what makes ordering matter.

## The curriculum

You can't hand an RL agent the hardest problem on day one, so programs are generated across five difficulty levels (`evaluator.py`). The agent generates its own training data, sets its own difficulty, and grades its own answers against the verifier.

| Level | Size | What it teaches |
|-------|------|-----------------|
| 1 | 4–6 instr | The basic constant-fold → dead-code-elimination loop |
| 2 | 8–12 instr | Chained arithmetic + a `LOAD` forced into the output (so you can't just fold everything to a constant) |
| 3 | 15–20 instr | Deeper chains, two `LOAD`s, optional second observable output |
| 4 | 20–28 instr | Fan-out topology — a few "hub" variables drive most of the arithmetic, so one good pass at a hub cascades through many dependents |
| 5 | 30–40 instr | Maximum density: heavy MUL/DIV, ~70% peephole bait (hidden 0s and 1s), multiple LOADs/STOREs |

Programs are random per seed, and the held-out evaluation set uses completely separate seeds so there's no leakage between training and test.

## Training

The policy is fine-tuned with **GRPO** (Group Relative Policy Optimization) through Hugging Face TRL. Instead of scoring one attempt at a time, the model generates several candidate pass-sequences for the same program and learns by comparing them against each other — it improves by watching its own decisions play out side by side. We used Unsloth with 4-bit quantization so the full loop fits on a single GPU.

One practical detail worth noting: LLMs are wordy. Asked for a JSON array of passes, the model will happily wrap it in "Certainly, based on my analysis…". So there's a regex fallback parser that ignores the prose, finds the brackets, and pulls the array out before it hits the environment. Small fix, big stability win.

> Heads up on installation order: torch, transformers, unsloth, and flash-attention are extremely opinionated about the order they get installed in. Pin versions, install deliberately, and restart the kernel between stages. See `requirements.txt`.

## Results

We built an `-O3`-style reference baseline — a fixed-point loop that keeps applying the forward passes until the cycle count stops moving — and evaluated on a 100-program held-out set.

- Baseline (forward fixed-point): **64.3% aggregate cycle reduction**
- Level 5 programs: roughly **84 → 26 cycles** on average after optimization
- Forward-only policies plateau in local minima; the reverse-pass policy is what reaches the global optimum

Training curves (reward, KL divergence, loss, action-sequence length) are in the `*.png` files in this repo.

## Running the demo

```bash
pip install -r requirements.txt
python app.py
```

This launches the Gradio UI. Pick a seed and curriculum level, choose up to three passes to apply in sequence, hit Run, and you'll see the per-step rewards, the cycle counts, and the final optimized pseudo-assembly. It's a smoke test for the environment more than a polished product — the point is to watch the passes (forward *and* reverse) act on a real program.

The demo is also deployed on Hugging Face Spaces (see the Spaces config README).

## Repo layout

```
toy_ir.py        # the VM, per-op cycle costs, and all five passes
evaluator.py     # the equivalence verifier + the 5-level curriculum generators
compiler_env.py  # CompilerTetrisEnv — the Gym-style reset/step wrapper
app.py           # Gradio demo
requirements.txt # GRPO / Unsloth / TRL training stack
*.png            # training curves
blog.md          # the longer story of how this got built
```

## What we'd do with more time

- Flesh out `peephole_optimization` with a real set of algebraic identities (×1, +0, ×0, ÷1, strength reduction).
- Widen the action space toward the full set of classic passes and let the agent reason over their interactions.
- Train and report on a larger, more varied program distribution, with the reverse passes available throughout.

The thing we're proudest of is simple: a system that learned not just how to make code faster, but *when to make it slower first* in order to get there. That second part is the whole game.
