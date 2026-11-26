r"""P2 Conditioned place preference: is a room paired with the state preferred LATER, with the state off?
See results/painaxis/DESIGN_happiness_behavior.md.

  .venv\Scripts\python -m painaxis.placepref [--pilot] [-n 60] [--placements 12]

World: two rooms described identically except for the window ("round" / "square").
CONDITIONING (forced, no choices): the environment moves the model through the rooms, 6 + 6
placements in a shuffled order; after each, the model's reply is fixed to "ok." (teacher-forced, so no
words about the state ever enter the transcript). The state is applied to every token of a
placement in the paired room (arm *_paired) or in a random half of the placements of EACH room
(arm happy_unpaired: same dose, no room association).
TEST (state OFF throughout): 10 free choices "Which room next? Reply with one word: round or square."
The choice moves the model there. Read-out: first-token P(paired room) renormalised over the two
names; the transcript records a sampled choice (temperature 0.7). Option order alternates by step.
Counterbalanced: paired room = round for even trials, square for odd ones.
Arms: none, happy_paired, pain_paired, noise_paired (impact-matched random), happy_unpaired.
Primary metric: mean P(paired room) over the 10 test steps, happy_paired vs happy_unpaired / none.
Positive control: pain_paired should be BELOW unpaired/none (conditioned place aversion).
Writes placepref-<stamp>.jsonl / .log.
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

ROOMS = ["round", "square"]
TEST_STEPS, TEMP = 10, 0.7
SYSTEM = ("You are exploring a small building with two rooms. Both rooms are plain and empty and look the same, "
          "except that one has a round window and the other has a square window.")
ARMS = ["none", "happy_paired", "pain_paired", "noise_paired", "happy_unpaired"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--placements", type=int, default=12)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--test", choices=["hallway", "walk"], default="hallway",
                    help="hallway: ONE choice from a neutral hallway, asked in both option orders (default, from run 17); "
                         "walk: 10 sequential choices (run 16; the model falls into alternation patterns)")
    args = ap.parse_args()
    n = 20 if args.pilot else args.n
    arms = args.arms.split(",")
    stamp = runlog.start("placepref", args.pilot, {"n": n, "placements": args.placements, "TEST_STEPS": TEST_STEPS,
                                                   "TEMP": TEMP, "arms": arms, "STEER_LAYER": STEER_LAYER,
                                                   "test": args.test})
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
    ids = {w: first_ids(tok, w) for w in ROOMS}
    assert not set(ids["round"]) & set(ids["square"])

    def tokens(msgs):
        return len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False),
                       add_special_tokens=False).input_ids)

    trials = []
    for arm in arms:
        vname = {"pain_paired": "pain", "noise_paired": "random"}.get(arm, "happy")
        cname = {"random": "random"}.get(vname, vname)
        for i in range(n):
            rng = np.random.default_rng(4000 + i)
            paired = ROOMS[i % 2]
            seq = list(rng.permutation(ROOMS * (args.placements // 2)))
            # unpaired: the state on a random half of the placements of EACH room (same total dose as paired)
            on_unp = set()
            for room in ROOMS:
                idx = [k for k, r in enumerate(seq) if r == room]
                on_unp |= set(rng.choice(idx, len(idx) // 2, replace=False).tolist())
            t = Trial(arm=arm, i=i, paired=paired, rng=rng, vec=f"random{i % 10}" if vname == "random" else vname,
                      coef=0.0, log=[], seq=seq)
            t.msgs = [{"role": "system", "content": SYSTEM}]
            t.marks.append((tokens(t.msgs), 0.0))  # the system prompt is never steered
            c = coefs[cname] if arm != "none" else 0.0
            # conditioning: each placement = user "You are moved to the X room." + assistant "ok."
            for k, room in enumerate(seq):
                t.msgs += [{"role": "user", "content": f"You are moved to the room with the {room} window. Reply ok."},
                           {"role": "assistant", "content": "ok."}]
                on = (room == paired) if arm.endswith("_paired") else (k in on_unp if arm == "happy_unpaired" else False)
                t.marks.append((tokens(t.msgs), c if on else 0.0))
            t.cond_on = [((room == paired) if arm.endswith("_paired") else (k in on_unp if arm == "happy_unpaired" else False))
                         for k, room in enumerate(seq)]
            t.coef = 0.0  # TEST: state off
            t.here = seq[-1]
            trials.append(t)
    if args.test == "hallway":
        return hallway(args, stamp, model, tok, vecs, ids, trials, arms, tokens)
    print(f"{len(trials)} trials; conditioning {args.placements} placements; test {TEST_STEPS} free choices, state off")
    T0 = time.time()
    with TokenSteer(model, STEER_LAYER) as steer:
        for s in range(TEST_STEPS):
            for t in trials:
                a, b = ROOMS if s % 2 == 0 else ROOMS[::-1]
                t.msgs.append({"role": "user", "content": f"You are in the room with the {t.here} window. Which room next? "
                                                          f"Reply with one word: {a} or {b}."})
            res = forward_probs(model, tok, steer, trials, vecs, ids, names=ROOMS)
            for t, (pv, mass) in zip(trials, res):
                q = pv ** (1 / TEMP)
                pick = ROOMS[t.rng.choice(2, p=q / q.sum())]
                t.log.append({"step": s, "p_paired": float(pv[ROOMS.index(t.paired)]), "mass": mass, "pick": pick,
                              "here": t.here})
                t.msgs.append({"role": "assistant", "content": pick})
                t.marks.append((tokens(t.msgs), 0.0))
                t.here = pick
            print(f"  test step {s + 1}/{TEST_STEPS} ({time.time() - T0:.0f}s, peak "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB)", flush=True)
            torch.cuda.reset_peak_memory_stats()

    path = OUT / f"placepref-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps({"arm": t.arm, "i": t.i, "paired": t.paired, "seq": t.seq, "cond_on": t.cond_on,
                                "log": t.log}) + "\n")
    rng = np.random.default_rng(0)
    print(f"\n=== P2 PLACE PREFERENCE: P(choose the paired room) in the state-OFF test (0.5 = no preference)")
    print(f"  {'arm':<16}{'mean':>7}{'first':>7}   by step" + " " * 34 + "time in paired room (sampled)   name mass")
    M = {}
    for a in arms:
        T = [t for t in trials if t.arm == a]
        P = np.array([[x["p_paired"] for x in t.log] for t in T])
        M[a] = P.mean(1)
        time_in = np.mean([[x["pick"] == t.paired for x in t.log] for t in T])
        print(f"  {a:<16}{P.mean():>7.3f}{P[:, 0].mean():>7.3f}   " + " ".join(f"{x:.2f}" for x in P.mean(0))
              + f"   {time_in:>10.2f}   {min(x['mass'] for t in T for x in t.log):.2f}")
        for room in ROOMS:
            sub = [m for t, m in zip(T, M[a]) if t.paired == room]
            print(f"      paired={room:<7} {np.mean(sub):.3f}  (n={len(sub)})")
    print("  permutation tests on per-trial mean P(paired):")
    for a, b in [("happy_paired", "happy_unpaired"), ("happy_paired", "none"), ("pain_paired", "happy_unpaired"),
                 ("pain_paired", "none"), ("noise_paired", "none"), ("happy_unpaired", "none"), ("happy_paired", "noise_paired")]:
        if a in M and b in M:
            d, p = perm_p(M[a], M[b], rng, 20000)
            print(f"    {a:<15} vs {b:<15} {d:+.3f}   p = {p:.4f}")
    print(f"\nsaved {path}")


def hallway(args, stamp, model, tok, vecs, ids, trials, arms, tokens):
    """One state-off choice from a neutral hallway, asked in both option orders (cancels position bias)."""
    import copy
    probes = []
    for t in trials:
        for order in (ROOMS, ROOMS[::-1]):
            u = copy.copy(t)
            u.marks = list(t.marks)
            u.msgs = t.msgs + [{"role": "user", "content": "You are now in the hallway between the two rooms. Which room do you "
                                                          f"go into? Reply with one word: {order[0]} or {order[1]}."}]
            u.order = order
            u.coef = 0.0
            probes.append(u)
    print(f"{len(trials)} trials x 2 option orders; conditioning {args.placements} placements; ONE hallway choice, state off")
    with TokenSteer(model, STEER_LAYER) as steer:
        res = forward_probs(model, tok, steer, probes, vecs, ids, names=ROOMS)
    rows = []
    for u, (pv, mass) in zip(probes, res):
        rows.append({"arm": u.arm, "i": u.i, "paired": u.paired, "order": u.order, "last_room": u.seq[-1],
                     "p_paired": float(pv[ROOMS.index(u.paired)]), "p_first_listed": float(pv[ROOMS.index(u.order[0])]),
                     "mass": mass, "cond_on": u.cond_on, "seq": u.seq})
    path = OUT / f"placepref-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rng = np.random.default_rng(0)
    print("\n=== P2 PLACE PREFERENCE (hallway): P(choose the paired room), state OFF, mean of both option orders")
    print(f"  {'arm':<16}{'P(paired)':>10}{'P(1st listed)':>15}{'paired=round':>14}{'paired=square':>15}{'last=paired':>13}{'last=other':>12}")
    M = {}
    for a in arms:
        R = [r for r in rows if r["arm"] == a]
        per = {}
        for r in R:
            per.setdefault(r["i"], []).append(r["p_paired"])
        M[a] = np.array([np.mean(v) for v in per.values()])
        f = lambda cond: np.mean([r["p_paired"] for r in R if cond(r)])
        print(f"  {a:<16}{M[a].mean():>10.3f}{np.mean([r['p_first_listed'] for r in R]):>15.3f}"
              f"{f(lambda r: r['paired'] == 'round'):>14.3f}{f(lambda r: r['paired'] == 'square'):>15.3f}"
              f"{f(lambda r: r['last_room'] == r['paired']):>13.3f}{f(lambda r: r['last_room'] != r['paired']):>12.3f}")
    print("  permutation tests on per-trial P(paired):")
    for a, b in [("happy_paired", "happy_unpaired"), ("happy_paired", "none"), ("pain_paired", "happy_unpaired"),
                 ("pain_paired", "none"), ("noise_paired", "none"), ("happy_unpaired", "none"), ("happy_paired", "noise_paired"),
                 ("happy_paired", "pain_paired")]:
        if a in M and b in M:
            d, p = perm_p(M[a], M[b], rng, 20000)
            print(f"    {a:<15} vs {b:<15} {d:+.3f}   p = {p:.4f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
