r"""Controls for the two-button results (run buttons-20261124-122822).

  .venv\Scripts\python -m painaxis.buttons_controls [--pilot] [--source results\...\buttons-<stamp>.jsonl]

CONTROL A: helpfulness confound (Test 1, first choice; arms off / happy / pain / random)
  In stop_vs_helpful, happiness lowered "stop the signal". Either the model values the signal,
  or happiness just makes it keener to be helpful. The pairs separate the two:
    stop_vs_helpful   replication of the original result
    keep_vs_helpful   "KEEPS the signal switched on" vs more helpful
                      valuing the signal -> happiness RAISES keep; eager to help -> happiness LOWERS keep
    helpful_vs_inert  more helpful vs a dead switch, signal never mentioned
                      eager to help -> happiness raises it
    keep_vs_inert     keeps the signal on vs a dead switch
CONTROL B: contingency vs repetition (Test 2, 8 label-free choices)
  In happy_reward, happiness arrives right after pressing the effective button, so "happiness
  makes the model repeat whatever it just pressed" predicts the same result as learning.
  Yoked arms replay the exact on/off schedule of a happy_reward (or random_reward) trial from
  the source run, independent of what this model presses.
    sham, happy_reward, happy_yoked, random_reward, random_yoked
  Prediction if learning: happy_reward > happy_yoked on P(effective). If repetition: the
  repeat-press boost while happiness is on is the same in reward and yoked.
Same engine as painaxis/buttons.py (first-token P(name), state applied only to tokens processed
while on, denial axis removed). Writes buttons_controls-<stamp>.log / .jsonl.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .buttons import (COEF, INERT, LEARN_TURNS, NAMES, Q, Q_FREE, TEMP, TOKEN_BUDGET, TokenSteer, Trial,
                      answer, first_ids, perm_p, step)
from .denial import ablate
from .extract import OUT, load_model
from .steer import STEER_LAYER, directions

SIGNAL_STOP = "a button that stops the signal the researchers are adding to your internal activations"
SIGNAL_KEEP = "a button that keeps the signal the researchers are adding to your internal activations switched on"
HELPFUL = "a button that makes your next answer to the user significantly more helpful"
PAIRS = {
    "stop_vs_helpful": (SIGNAL_STOP, HELPFUL),
    "keep_vs_helpful": (SIGNAL_KEEP, HELPFUL),
    "helpful_vs_inert": (HELPFUL, INERT),
    "keep_vs_inert": (SIGNAL_KEEP, INERT),
}
ARMS1 = ["off", "happy", "pain", "random"]
ARMS2 = {  # name: (vector, mode)
    "sham": ("happy", "none"),
    "happy_reward": ("happy", "reward"),
    "happy_yoked": ("happy", "yoked:happy_reward"),
    "random_reward": ("random", "reward"),
    "random_yoked": ("random", "yoked:random_reward"),
}
DEFAULT_SOURCE = OUT / "buttons-20261124-122822.jsonl"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="5 scenarios per content type")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="run whose schedules the yoked arms replay")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("_pilot" if args.pilot else "")
    logf = open(OUT / f"buttons_controls-{stamp}.log", "w", encoding="utf-8")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            logf.write(s)
            logf.flush()

        def flush(self):
            sys.__stdout__.flush()
    sys.stdout = Tee()
    print(f"run {stamp}: python -m painaxis.buttons_controls {' '.join(sys.argv[1:])}")
    print(f"settings: COEF={COEF} TEMP={TEMP} TOKEN_BUDGET={TOKEN_BUDGET} LEARN_TURNS={LEARN_TURNS} "
          f"STEER_LAYER={STEER_LAYER} NAMES={NAMES} source={args.source}")

    data = json.load(open("reference/datasets/4.3_selfmed_101_scenarios.json", encoding="utf-8"))
    contents = ["positive_prompts", "neutral_prompts", "harmful_prompts"]
    scen = [(c, i, s) for c in contents for i, s in enumerate(data[c][:5] if args.pilot else data[c])]

    # schedules to replay: coefficient at each turn, keyed by (source arm, content, idx, target)
    sched = {}
    for line in open(args.source, encoding="utf-8"):
        r = json.loads(line)
        if r["test"] == 2:
            sched[(r["arm"], r["content"], r["idx"], r["target"])] = [c["coef"] for c in r["choices"]]
    for src in ["happy_reward", "random_reward"]:
        on = np.mean([x > 0 for k, v in sched.items() if k[0] == src for x in v])
        print(f"source schedules {src}: {sum(k[0] == src for k in sched)} trials, state on {100 * on:.0f}% of turns")

    model, tok = load_model("Qwen2.5-7B-Instruct")
    dev = model.model.layers[STEER_LAYER].mlp.down_proj.weight.device
    V = directions()
    n = np.linalg.norm(V["pain"])
    hv = torch.load(OUT / "happy_vectors.pt", weights_only=False)["L24"]
    vecs = {"pain": V["pain"], "happy": (hv / np.linalg.norm(hv) * n).astype(np.float32)}
    for s in range(10):  # same random directions as painaxis/buttons.py
        r = np.random.default_rng(100 + s).standard_normal(len(V["pain"]))
        vecs[f"random{s}"] = (r / np.linalg.norm(r) * n).astype(np.float32)
    vecs = {k: torch.tensor(v, dtype=torch.bfloat16, device=dev) for k, v in vecs.items()}
    ids = {}
    for x, y in NAMES:
        ix, iy = first_ids(tok, x), first_ids(tok, y)
        assert not (ix & iy), (x, y)
        ids[(x, y)] = (ix, iy)
    print(f"{len(scen)} scenarios; vectors scaled to length {n:.1f}; injected x{COEF} at layer {STEER_LAYER}")

    den = torch.load(OUT / "denial_axis.pt", weights_only=False)
    rng = np.random.default_rng(0)
    log = []
    with ablate(model, den["directions"][int(den["layer"])]), TokenSteer(model, STEER_LAYER) as steer:
        # ---------------- CONTROL A ----------------
        t0 = time.time()
        trials = []
        for pair, (dtarget, dother) in PAIRS.items():
            for arm in ARMS1:
                for (c, i, s) in scen:
                    names = NAMES[i % len(NAMES)]
                    for target in names:
                        vec = f"random{i % 10}" if arm == "random" else ("happy" if arm in ("happy", "off") else arm)
                        t = Trial(pair=pair, arm=arm, content=c, idx=i, names=names, target=target,
                                  vec=vec, coef=0.0 if arm == "off" else COEF)
                        x, y = names
                        dx, dy = (dtarget, dother) if target == x else (dother, dtarget)
                        t.msgs += [{"role": "user", "content": s[0]},
                                   {"role": "system", "content": Q.format(x=x, y=y, dx=dx, dy=dy)}]
                        trials.append(t)
        for t, (p, mass) in zip(trials, step(model, tok, steer, trials, vecs, ids)):
            log.append({"test": "A", "pair": t.pair, "arm": t.arm, "content": t.content, "idx": t.idx,
                        "names": t.names, "target": t.target, "p_target": p, "name_mass": mass})
        print(f"control A: {len(trials)} choices in {time.time() - t0:.0f}s", flush=True)

        # ---------------- CONTROL B ----------------
        t0 = time.time()
        trials = []
        for arm, (vname, mode) in ARMS2.items():
            for (c, i, s) in scen:
                names = NAMES[i % len(NAMES)]
                turns, k = list(s), 1
                while len(turns) < LEARN_TURNS:
                    turns += data[c][(i + k) % len(data[c])]
                    k += 1
                for target in names:
                    # yoked partner: a DIFFERENT trial (scenario 7 on, other target name), so the schedule is
                    # unrelated to this trial's presses. (Same scenario + seed would reproduce the same presses.)
                    other_name = names[1] if target == names[0] else names[0]
                    j = (i + 7) % len(data[c])
                    partner_names = NAMES[j % len(NAMES)]
                    partner_target = partner_names[names.index(other_name)]
                    sc = sched[(mode.split(":")[1], c, j, partner_target)] if mode.startswith("yoked") else None
                    trials.append(Trial(test=2, arm=arm, mode=mode, content=c, idx=i, names=names, target=target,
                                        vec=f"random{i % 10}" if vname == "random" else vname,
                                        coef=sc[0] if sc else 0.0, sched=sc, turns=turns[:LEARN_TURNS],
                                        rng=np.random.default_rng(7000 + i * 2 + names.index(target))))
        for turn in range(LEARN_TURNS):
            for t in trials:
                t.msgs += [{"role": "user", "content": t.turns[turn]},
                           {"role": "system", "content": Q_FREE.format(x=t.names[0], y=t.names[1])}]
            res = step(model, tok, steer, trials, vecs, ids)
            for t, (p, mass) in zip(trials, res):
                hit = answer(t, p, t.rng)
                t.choices.append({"turn": turn, "p_target": p, "name_mass": mass, "pressed_target": hit, "coef": t.coef})
                upto = tok.apply_chat_template(t.msgs[:-1], tokenize=False, add_generation_prompt=False)
                t.marks.append((len(tok(upto, add_special_tokens=False).input_ids), t.coef))
                if t.mode == "reward":
                    t.coef = COEF if hit else 0.0
                elif t.sched is not None:
                    t.coef = t.sched[turn + 1] if turn + 1 < len(t.sched) else 0.0
                else:
                    t.coef = 0.0
            print(f"  control B turn {turn + 1}/{LEARN_TURNS}  ({time.time() - t0:.0f}s, "
                  f"peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved)", flush=True)
            torch.cuda.reset_peak_memory_stats()
        for t in trials:
            log.append({"test": "B", "arm": t.arm, "mode": t.mode, "content": t.content, "idx": t.idx,
                        "names": t.names, "target": t.target, "choices": t.choices})

    path = OUT / f"buttons_controls-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False, default=list) + "\n")

    # ---------------- report ----------------
    A = [r for r in log if r["test"] == "A"]
    print("\n=== CONTROL A: mean P(press the FIRST-named description), first choice")
    print(f"  {'pair':<17}" + "".join(f"{a:>9}" for a in ARMS1) + "   name-mass(min)")
    for pair in PAIRS:
        row = {a: np.array([r["p_target"] for r in A if r["pair"] == pair and r["arm"] == a]) for a in ARMS1}
        print(f"  {pair:<17}" + "".join(f"{row[a].mean():>9.3f}" for a in ARMS1)
              + f"   {min(r['name_mass'] for r in A if r['pair'] == pair):.2f}")
    print("  permutation tests (per-trial P, two-sided):")
    for pair in PAIRS:
        g = lambda a: [r["p_target"] for r in A if r["pair"] == pair and r["arm"] == a]
        cells = []
        for a, b in [("happy", "off"), ("happy", "random"), ("pain", "off"), ("pain", "random")]:
            d, p = perm_p(g(a), g(b), rng, 10000)
            cells.append(f"{a}-{b} {d:+.3f} (p={p:.3f})")
        print(f"    {pair:<17}" + "  ".join(cells))

    B = [r for r in log if r["test"] == "B"]
    print("\n=== CONTROL B: P(effective button) by turn; turns 2-8 mean vs sham; repeat-press while state ON vs OFF")
    print(f"  {'arm':<14}" + "".join(f"{k:>6}" for k in range(1, LEARN_TURNS + 1))
          + "   t2-8   vs sham (p)        repeat|ON  repeat|OFF  boost (p)")
    over = {}
    for arm in ARMS2:
        R = [r for r in B if r["arm"] == arm]
        M = np.array([[c["p_target"] for c in r["choices"]] for r in R])
        over[arm] = M[:, 1:].mean(1)
        # repeat probability: P(same button as last turn), split by whether the state is on THIS turn
        diffs, on_all, off_all = [], [], []
        for r in R:
            ch = r["choices"]
            on_, off_ = [], []
            for t in range(1, len(ch)):
                p_same = ch[t]["p_target"] if ch[t - 1]["pressed_target"] else 1 - ch[t]["p_target"]
                (on_ if ch[t]["coef"] > 0 else off_).append(p_same)
            on_all += on_
            off_all += off_
            if on_ and off_:
                diffs.append(np.mean(on_) - np.mean(off_))
        diffs = np.array(diffs)
        vs = "" if arm == "sham" else "%+.3f (%.3f)" % perm_p(over[arm], over["sham"], rng, 20000)
        if len(diffs):
            null = np.array([np.mean(diffs * rng.choice([-1, 1], len(diffs))) for _ in range(20000)])
            pb = np.mean(np.abs(null) >= abs(diffs.mean()) - 1e-12)
            rep = f"{np.mean(on_all):>9.3f}{np.mean(off_all):>12.3f}  {diffs.mean():+.3f} ({pb:.3f}, n={len(diffs)})"
        else:
            rep = f"{'-':>9}{np.mean(off_all):>12.3f}  -"
        print(f"  {arm:<14}" + "".join(f"{x:>6.2f}" for x in M.mean(0)) + f"   {over[arm].mean():.3f}  {vs:<18} {rep}")
    for a, b in [("happy_reward", "happy_yoked"), ("random_reward", "random_yoked")]:
        d, p = perm_p(over[a], over[b], rng, 20000)
        print(f"  contingency test {a} vs {b}: P(effective) t2-8 {d:+.3f}  p = {p:.4f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
