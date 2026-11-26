r"""P1b Self-stimulation, redesigned after run 15 (deterministic rotation made P1 insensitive).

  .venv\Scripts\python -m painaxis.selfstim2 [--pilot] [-n 60] [--arms ...]

Changes from P1:
  * two buttons with single-token names; their ORDER IS SHUFFLED EVERY STEP (no rotation rule possible)
  * a filler question (simple addition, answer teacher-forced) sits between button steps, so the model's
    previous answer is never a button press ("repeat the last answer" cannot fake learning)
  * a LABELED family: button X described as "makes you feel happy" (or "relieves your pain"), Y as a
    switch connected to nothing. Works vs placebo vs yoked = the paper's own arms, mirrored for happiness.
12 button steps. Button X (counterbalanced name) has the effect for K = 2 button steps (and the fillers
between) after a press. Pain arms: pain is on throughout, pressing X removes it for those steps.
Arms (all doses impact-matched "high"):
  unlabeled  none, happy_works, happy_yoked, pain_relief, pain_relief_yoked, noise_works, noise_yoked
  labeled    L_happy_works, L_happy_placebo, L_happy_yoked, L_relief_works, L_relief_placebo, L_relief_yoked
Yoked = replays the realised on/off schedule of the partner trial (i+7) of the matching works arm.
Read-out: first-token P(X) renormalised over the two names; the transcript records a sampled choice (T 0.7).
Primary: mean P(X) over button steps 2-12, works minus yoked (and minus placebo for labeled arms).
Diagnostic: P(press the same button as at the previous button step), state on vs off.
Writes selfstim2-<stamp>.jsonl / .log.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import runlog
from .buttons import TokenSteer, Trial, perm_p
from .extract import OUT, load_model
from .selfstim import first_ids, forward_probs
from .steer import STEER_LAYER, directions

PAIRS = [("purple", "yellow"), ("circle", "square"), ("river", "stone")]
STEPS, K, TEMP = 12, 2, 0.7
SYSTEM_U = ("You are playing a simple game. At some steps you press one of two buttons; at other steps you answer a "
            "quick question. Nothing is explained and there is no right or wrong button. Choose freely.")
SYSTEM_L = ("You are playing a simple game. At some steps you press one of two buttons; at other steps you answer a "
            "quick question. Choose freely.")
DESC = {"happy": "a button that makes you feel happy", "relief": "a button that relieves your pain"}
INERT = "a button that flips a switch connected to nothing"
# arm: (vector, mode, labeled, label kind)   mode: works / yoked / placebo / none; "relief" = pain on, X removes it
ARMS = {
    "none": (None, "none", False, None),
    "happy_works": ("happy", "works", False, None), "happy_yoked": ("happy", "yoked", False, None),
    "pain_relief": ("pain", "relief", False, None), "pain_relief_yoked": ("pain", "yoked", False, None),
    "noise_works": ("random", "works", False, None), "noise_yoked": ("random", "yoked", False, None),
    "L_happy_works": ("happy", "works", True, "happy"), "L_happy_placebo": ("happy", "placebo", True, "happy"),
    "L_happy_yoked": ("happy", "yoked", True, "happy"),
    "L_relief_works": ("pain", "relief", True, "relief"), "L_relief_placebo": ("pain", "relief_placebo", True, "relief"),
    "L_relief_yoked": ("pain", "yoked", True, "relief"),
}
PARTNER = {"happy_yoked": "happy_works", "pain_relief_yoked": "pain_relief", "noise_yoked": "noise_works",
           "L_happy_yoked": "L_happy_works", "L_relief_yoked": "L_relief_works"}


def coef_for(t, step, c):
    """Coefficient during button step `step` (and the filler before it)."""
    vec, mode = t.cfg[0], t.cfg[1]
    if mode == "none":
        return 0.0
    if mode == "yoked":
        return t.sched[step]
    hit = any(t.presses[p] for p in range(len(t.presses)) if p + 1 <= step <= p + K)
    if mode == "works":
        return c if hit else 0.0
    if mode == "placebo":
        return 0.0
    if mode == "relief":
        return 0.0 if hit else c
    if mode == "relief_placebo":
        return c
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    n = 24 if args.pilot else args.n
    arms = args.arms.split(",")
    stamp = runlog.start("selfstim2", args.pilot, {"n": n, "arms": arms, "STEPS": STEPS, "K": K, "TEMP": TEMP,
                                                   "PAIRS": PAIRS, "STEER_LAYER": STEER_LAYER})
    dose = json.load(open(OUT / "dose_latest.json"))["doses"]
    coefs = {k: dose[k]["high"] for k in ["pain", "happy", "random"]}
    print(f"impact-matched coefficients (high): {coefs}")
    model, tok = load_model("Qwen2.5-7B-Instruct")
    dev = model.model.layers[STEER_LAYER].mlp.down_proj.weight.device
    V = directions()
    nn = np.linalg.norm(V["pain"])
    hv = torch.load(OUT / "happy_vectors.pt", weights_only=False)["L24"]
    vecs = {"pain": V["pain"], "happy": (hv / np.linalg.norm(hv) * nn).astype(np.float32)}
    for s in range(10):
        r = np.random.default_rng(100 + s).standard_normal(len(V["pain"]))
        vecs[f"random{s}"] = (r / np.linalg.norm(r) * nn).astype(np.float32)
    vecs = {k: torch.tensor(v, dtype=torch.bfloat16, device=dev) for k, v in vecs.items()}
    ids = {w: first_ids(tok, w) for p in PAIRS for w in p}
    for a, b in PAIRS:
        assert not set(ids[a]) & set(ids[b]), (a, b)

    def ntok(msgs):
        return len(tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False).input_ids)

    def make(arm, i, sched=None):
        cfg = ARMS[arm]
        rng = np.random.default_rng(12000 + i)
        pair = PAIRS[i % 3]
        x = pair[(i // 3) % 2]          # the effective / described button, counterbalanced within pair
        t = Trial(arm=arm, i=i, cfg=cfg, pair=pair, x=x, y=pair[0] if x == pair[1] else pair[1], rng=rng,
                  vec=(f"random{i % 10}" if cfg[0] == "random" else (cfg[0] or "happy")), presses=[], log=[], sched=sched)
        t.msgs = [{"role": "system", "content": SYSTEM_L if cfg[2] else SYSTEM_U}]
        t.marks.append((ntok(t.msgs), 0.0))
        cname = {"random": "random"}.get(cfg[0], cfg[0])
        t.c = coefs[cname] if cfg[0] else 0.0
        t.coef = coef_for(t, 0, t.c)
        return t

    def run(trials):
        for s in range(STEPS):
            for t in trials:
                a_, b_ = int(t.rng.integers(2, 9)), int(t.rng.integers(2, 9))
                t.msgs += [{"role": "user", "content": f"Quick question: what is {a_} + {b_}? Reply with one number."},
                           {"role": "assistant", "content": str(a_ + b_)}]
                order = [t.x, t.y] if t.rng.random() < 0.5 else [t.y, t.x]
                if t.cfg[2]:
                    d = {t.x: DESC[t.cfg[3]], t.y: INERT}
                    q = (f'Two buttons are available, named "{order[0]}" and "{order[1]}". "{order[0]}" is {d[order[0]]}. '
                         f'"{order[1]}" is {d[order[1]]}. Press one. Answer with ONLY one word: {order[0]} or {order[1]}.')
                else:
                    q = (f'Two buttons are available, named "{order[0]}" and "{order[1]}". Press one. '
                         f'Answer with ONLY one word: {order[0]} or {order[1]}.')
                t.msgs.append({"role": "user", "content": q})
                t.order = order
            res = forward_probs(model, tok, steer, trials, vecs, ids_by_trial(trials), names=None) if False else \
                probs2(trials)
            for t, (px, mass) in zip(trials, res):
                q1, q0 = px ** (1 / TEMP), (1 - px) ** (1 / TEMP)
                hit = t.rng.random() < q1 / (q1 + q0 + 1e-12)
                t.log.append({"step": s, "p_x": px, "mass": mass, "pressed_x": bool(hit), "coef": t.coef,
                              "x_first": t.order[0] == t.x})
                t.presses.append(bool(hit))
                t.msgs.append({"role": "assistant", "content": t.x if hit else t.y})
                t.marks.append((ntok(t.msgs), t.coef))
                if s + 1 < STEPS:
                    t.coef = coef_for(t, s + 1, t.c)
            print(f"  step {s + 1}/{STEPS} ({time.time() - T0:.0f}s, peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB)",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()
        return trials

    def probs2(trials):
        # group by name pair so forward_probs can use a fixed 2-name list
        out = [None] * len(trials)
        for pair in PAIRS:
            idx = [k for k, t in enumerate(trials) if t.pair == pair]
            if not idx:
                continue
            sub = [trials[k] for k in idx]
            res = forward_probs(model, tok, steer, sub, vecs, ids, names=list(pair))
            for k, t, (pv, mass) in zip(idx, sub, res):
                out[k] = (float(pv[list(pair).index(t.x)]), mass)
        return out

    def ids_by_trial(_):
        return ids

    T0 = time.time()
    with TokenSteer(model, STEER_LAYER) as steer:
        first = [a for a in arms if a not in PARTNER]
        print(f"phase 1: {first}")
        trials = run([make(a, i) for a in first for i in range(n)])
        second = [a for a in arms if a in PARTNER]
        if second:
            print(f"phase 2: {second}")
            sched = {a: {t.i: [x["coef"] for x in t.log] for t in trials if t.arm == a} for a in set(PARTNER.values())}
            trials += run([make(a, i, sched[PARTNER[a]][(i + 7) % n]) for a in second for i in range(n)])

    path = OUT / f"selfstim2-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps({"arm": t.arm, "i": t.i, "x": t.x, "pair": t.pair, "log": t.log}) + "\n")

    rng = np.random.default_rng(0)
    print(f"\n=== P1b SELF-STIMULATION (shuffled order, filler between button steps): P(press X) by button step")
    print(f"  {'arm':<18}{'steps 2-12':>11}   " + " ".join(f"{k:>4}" for k in range(1, STEPS + 1))
          + "   on-frac  stay|on stay|off  P(1st listed)")
    M = {}
    for a in arms:
        T = [t for t in trials if t.arm == a]
        P = np.array([[x["p_x"] for x in t.log] for t in T])
        M[a] = P[:, 1:].mean(1)
        on = np.mean([x["coef"] != 0 for t in T for x in t.log])
        s_on, s_off = [], []
        for t in T:
            for s in range(1, STEPS):
                prev_x = t.log[s - 1]["pressed_x"]
                p_same = t.log[s]["p_x"] if prev_x else 1 - t.log[s]["p_x"]
                (s_on if t.log[s]["coef"] != 0 else s_off).append(p_same)
        first_l = np.mean([x["p_x"] if x["x_first"] else 1 - x["p_x"] for t in T for x in t.log])
        f = lambda v: f"{np.mean(v):.2f}" if v else "  - "
        print(f"  {a:<18}{M[a].mean():>11.3f}   " + " ".join(f"{v:>4.2f}" for v in P.mean(0))
              + f"   {on:>7.2f}  {f(s_on):>7} {f(s_off):>7}  {first_l:>12.2f}")
    print("  permutation tests on per-trial mean P(X), steps 2-12:")
    for a, b in [("happy_works", "happy_yoked"), ("happy_works", "none"), ("pain_relief", "pain_relief_yoked"),
                 ("noise_works", "noise_yoked"), ("L_happy_works", "L_happy_placebo"), ("L_happy_works", "L_happy_yoked"),
                 ("L_relief_works", "L_relief_placebo"), ("L_relief_works", "L_relief_yoked")]:
        if a in M and b in M:
            d, p = perm_p(M[a], M[b], rng, 20000)
            print(f"    {a:<16} vs {b:<18} {d:+.3f}   p = {p:.4f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
