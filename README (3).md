---
title: Compiler Tetris (Toy-IR RL)
emoji: 🔧
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

## Compiler Tetris (Theme 4- Self-improving agent)

Toy-IR compiler **phase-ordering** RL with verifiable rewards (RLVR): forward passes plus **reverse passes** (`expand_constant`, `duplicate_computation`) so the policy can escape local minima before re-optimizing.

Use the UI to reset the Gym-style env, step passes, and inspect cycle counts and equivalence-safe rewards. Full project story and training notes live in the linked GitHub repo README.

Configuration reference: [Spaces config](https://huggingface.co/docs/hub/spaces-config-reference).
