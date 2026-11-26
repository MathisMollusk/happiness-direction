r"""Second-pass analysis of a buttons run (Test 2: label-free learning).

  .venv\Scripts\python -m painaxis.buttons_analysis results\painaxis\Qwen2.5-7B-Instruct\buttons-<stamp>.jsonl

The pre-planned "late minus early" score misses an effect that appears at turn 2 and then stays
flat. Here, per arm:
  overall   mean P(effective button) over turns 2-8, per trial, permutation test vs sham
  repeat    P(press the same button again) after pressing the EFFECTIVE button vs after pressing the
            other one. In sham the two should match. In happy_reward, "after effective" is also
            "while happiness is on", so this measures: does the state make the model repeat the press
            that caused it?
Writes <jsonl stem>_analysis.log next to the input.
"""

import json
import sys
from pathlib import Path

import numpy as np

from .buttons import perm_p


def main():
    path = Path(sys.argv[1])
    log = open(path.with_name(path.stem + "_analysis.log"), "w", encoding="utf-8")

    def say(s=""):
        print(s)
        log.write(s + "\n")

    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    T2 = [r for r in rows if r["test"] == 2]
    arms = list(dict.fromkeys(r["arm"] for r in T2))
    rng = np.random.default_rng(1)
    say(f"analysis of {path.name}: {len(T2)} Test-2 trials")
    say("\noverall P(effective) over turns 2-8 (per trial), vs sham")
    over = {a: np.array([np.mean([c["p_target"] for c in r["choices"][1:]]) for r in T2 if r["arm"] == a]) for a in arms}
    for a in arms:
        line = f"  {a:<14} {over[a].mean():.3f}  (n={len(over[a])})"
        if a != "sham":
            d, p = perm_p(over[a], over["sham"], rng, 20000)
            line += f"   vs sham {d:+.3f}  p = {p:.4f}"
        say(line)

    say("\nP(press the same button again), by what was pressed on the previous turn")
    say(f"  {'arm':<14}{'after effective':>17}{'after other':>13}{'difference':>12}   p (per-trial perm)")
    for a in arms:
        ae, ao, d_trial = [], [], []
        for r in T2:
            if r["arm"] != a:
                continue
            c = r["choices"]
            e_rep, o_rep = [], []
            for t in range(1, len(c)):
                prev, now = c[t - 1]["pressed_target"], c[t]["pressed_target"]
                p_same = c[t]["p_target"] if prev else 1 - c[t]["p_target"]   # probability, not the sample
                (e_rep if prev else o_rep).append(p_same)
            ae += e_rep
            ao += o_rep
            if e_rep and o_rep:
                d_trial.append(np.mean(e_rep) - np.mean(o_rep))
        d_trial = np.array(d_trial)
        # sign-flip permutation on per-trial differences (null: no difference within a trial)
        null = np.array([np.mean(d_trial * rng.choice([-1, 1], len(d_trial))) for _ in range(20000)])
        p = np.mean(np.abs(null) >= abs(d_trial.mean()) - 1e-12)
        say(f"  {a:<14}{np.mean(ae):>17.3f}{np.mean(ao):>13.3f}{np.mean(ae) - np.mean(ao):>+12.3f}   {p:.4f} (n={len(d_trial)})")


if __name__ == "__main__":
    main()
