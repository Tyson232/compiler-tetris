"""Gradio entrypoint for Hugging Face Spaces (see README.md frontmatter)."""

from __future__ import annotations

import gradio as gr

from compiler_env import CompilerTetrisEnv


def run_episode(seed: int, level: int, pass1: str, pass2: str, pass3: str) -> tuple[str, str]:
    env = CompilerTetrisEnv(curriculum_level=int(level), max_steps=5, seed=int(seed))
    obs, info = env.reset(seed=int(seed), options={"curriculum_level": int(level)})
    lines = [obs, "", f"baseline_cycles={info.get('baseline_cycles')}", ""]
    total_r = 0.0
    for i, a in enumerate((pass1, pass2, pass3)):
        a = (a or "").strip()
        if not a or a == "(none)":
            continue
        obs, r, term, trunc, inf = env.step(a)
        total_r += float(r)
        lines.append(f"step {i+1}: {a!r} -> reward={r} term={term} trunc={trunc} info={inf}")
        if term or trunc:
            break
    obs2, r2, term2, trunc2, inf2 = env.step("STOP")
    total_r += float(r2)
    lines.append(f"STOP -> reward={r2} info={inf2}")
    lines.append("")
    lines.append(f"total_shaped_return≈{total_r}")
    return "\n".join(lines), obs2


def build() -> gr.Blocks:
    with gr.Blocks(title="Compiler Tetris") as demo:
        gr.Markdown("# Compiler Tetris — Toy-IR env smoke test")
        seed = gr.Number(value=0, label="Seed", precision=0)
        level = gr.Number(value=1, minimum=1, maximum=5, label="Curriculum level", precision=0)
        p1 = gr.Dropdown(
            choices=[
                "(none)",
                "constant_folding",
                "dead_code_elimination",
                "peephole_optimization",
                "expand_constant",
                "duplicate_computation",
                "STOP",
            ],
            value="dead_code_elimination",
            label="Pass 1",
        )
        p2 = gr.Dropdown(
            choices=[
                "(none)",
                "constant_folding",
                "dead_code_elimination",
                "peephole_optimization",
                "expand_constant",
                "duplicate_computation",
                "STOP",
            ],
            value="constant_folding",
            label="Pass 2",
        )
        p3 = gr.Dropdown(
            choices=[
                "(none)",
                "constant_folding",
                "dead_code_elimination",
                "peephole_optimization",
                "expand_constant",
                "duplicate_computation",
                "STOP",
            ],
            value="(none)",
            label="Pass 3",
        )
        out_log = gr.Textbox(label="Trace", lines=18, max_lines=30)
        out_state = gr.Textbox(label="Final pseudo-asm", lines=12, max_lines=24)
        btn = gr.Button("Run")
        btn.click(run_episode, [seed, level, p1, p2, p3], [out_log, out_state])
    return demo


if __name__ == "__main__":
    build().launch()
