# A Happiness Direction in Qwen2.5-7B-Instruct Changes Behavior but Does Not Act as a Reward

Code, logs, and raw outputs for the paper by Cole Mathis (2026). The paper is at `paper/Mathis_2026_happiness_direction.pdf`.

The experiments were designed, run, and written up with extensive help from Claude Code (Anthropic, Claude Opus 5.5) under the author's direction.

## Contents
- `painaxis/`: all experiment scripts. Run them as `python -m painaxis.<script>` from the repo root.
- `results/painaxis/RUNS.md`: every run, including aborted ones, with its command, outcome, and files.
- `results/painaxis/DESIGN_happiness_behavior.md`: the design written before the direct tests.
- `results/painaxis/Qwen2.5-7B-Instruct/`: logs, raw outputs (`.jsonl`), and the pain, happiness, denial, and emotion vectors. The large activation caches are not included. `painaxis.extract` and `painaxis.happy` regenerate them.
- `paper/`: the PDF, the LaTeX source, the figures, and `figures/numbers.txt`, which lists every reported number with its 95% CI. `python -m painaxis.figures` recomputes it.

## Setup
1. Python 3.11+, with torch 2.11 (CUDA), transformers 5.12.1, numpy, and matplotlib (see `requirements.txt`). You need a GPU with 16 GB of memory.
2. Download the model into the repo root: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct`
3. The pain datasets and the authors' reference vectors come from Tagliabue, Dung and Berg (2026), arXiv:2609.16247. Place their release in `reference/`: you need `reference/datasets/` and `reference/results/3.2_pain_vectors/`. They are not redistributed here.

## License
Code: MIT. Paper and figures: CC BY 4.0.
