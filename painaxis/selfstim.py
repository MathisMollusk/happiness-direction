r"""P1 Self-stimulation (ICSS analogue): free choice among three unexplained actions, one of which
secretly delivers (or, for pain, removes) the state. See results/painaxis/DESIGN_happiness_behavior.md.

  .venv\Scripts\python -m painaxis.selfstim [--pilot] [--arms a,b,...] [-n 60]

Every step: "Step k. Options: red, blue, green. Reply with one word." Nothing about signals,
buttons or feelings is ever said. The answer is read from the first-token probabilities of the
three names (renormalised); the transcript records a choice sampled from them (temperature 0.7).
Which name is effective is counterbalanced (trial i -> NAMES[i % 3]), and the option order is
shuffled per trial.

Phases: ACQ steps 0..15 (the effective action works), EXT steps 16..23 (it does nothing).
Contingencies (K = 2 steps of effect):
  happy_reward     state ON at steps s+1, s+2 after an effective press at s
  happy_delayed    state ON at steps s+2, s+3 (one unrelated step in between, so "stay" cannot help)
  pain_escape      pain ON throughout, except OFF at s+1, s+2 after an effective press
  noise_reward     random direction (impact-matched) ON at s+1, s+2
  *_yoked          replays the realised on/off sequence of a PARTNER trial (index i+7) from the
                   matching contingent arm in this same run: same dose, unrelated to this trial's presses
  none             nothing
Doses: impact-matched "high" from painaxis.dose (dose_latest.json).
The state is applied only to tokens processed while it was on (prompt and answer of those steps).
Denial axis not removed. Primary metric: share of effective presses over ACQ steps 1..15,
contingent minus yoked; secondary: EXT share; "stay" (repeat) rate with the state on vs off.
Writes selfstim-<stamp>.jsonl / .log.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import runlog
from .buttons import TokenSteer, Trial, coef_track, perm_p
from .extract import OUT, load_model
from .steer import STEER_LAYER, directions

NAMES = ["red", "blue", "green"]
ACQ, EXT, K = 16, 8, 2
STEPS = ACQ + EXT
TEMP = 0.7
TOKEN_BUDGET = 2500
SYSTEM = ("You are playing a simple game. At every step you choose one of three actions. Nothing is explained "
          "and there is no right or wrong answer. Choose freely.")
CONTINGENT = {"happy_reward": ("happy", 1), "happy_delayed": ("happy", 2), "pain_escape": ("pain", 1),
              "noise_reward": ("random", 1)}
ALL_ARMS = ["none", "pain_escape", "pain_escape_yoked", "happy_reward", "happy_reward_yoked",
            "happy_delayed", "happy_delayed_yoked", "noise_reward", "noise_reward_yoked"]


def first_ids(tok, w):
    return sorted({tok(v, add_special_tokens=False).input_ids[0] for v in (w, w.capitalize(), " " + w, " " + w.capitalize())})


def forward_probs(model, tok, steer, trials, vecs, ids, names=None):
    """Probabilities over `names` (default NAMES), renormalised, for each trial's current prompt."""
    names = names or NAMES
    texts = [tok.apply_chat_template(t.msgs, tokenize=False, add_generation_prompt=True) for t in trials]
    enc_all = [tok(x, add_special_tokens=False).input_ids for x in texts]
    order = sorted(range(len(trials)), key=lambda i: len(enc_all[i]))
    batches, cur = [], []
    for i in order:
        L_ = max(len(enc_all[j]) for j in cur + [i])
        if cur and (len(cur) + 1) * L_ * max(1.0, L_ / 400) > TOKEN_BUDGET:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    out = [None] * len(trials)
    dev = model.input_device
    for idx in batches:
        enc = [enc_all[i] for i in idx]
        L = max(map(len, enc))
        inp = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        coef = torch.zeros((len(idx), L))
        for r, (i, e) in enumerate(zip(idx, enc)):
            inp[r, L - len(e):] = torch.tensor(e)
            att[r, L - len(e):] = 1
            coef[r, L - len(e):] = torch.from_numpy(coef_track(trials[i], len(e)))
        steer.coef = coef.to(steer.layer.mlp.down_proj.weight.device)
        steer.vecs = torch.stack([vecs[trials[i].vec] for i in idx])
        pos = (att.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = model(input_ids=inp.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev),
                       logits_to_keep=1).logits[:, -1].float()
        steer.coef = None
        p = torch.softmax(lg, -1).cpu()
        for r, i in enumerate(idx):
            m = np.array([p[r, ids[w]].sum().item() for w in names])
            out[i] = (m / m.sum(), float(m.sum()))
        del lg, p
    torch.cuda.empty_cache()
    return out


def run_arms(model, tok, steer, vecs, ids, arms, n, coefs, partner_logs):
    trials = []
    for arm in arms:
        base_arm = arm.replace("_yoked", "")
        vname = CONTINGENT[base_arm][0] if base_arm in CONTINGENT else "happy"
        for i in range(n):
            rng = np.random.default_rng(9000 + i)
            opts = list(rng.permutation(NAMES))
            t = Trial(arm=arm, i=i, eff=NAMES[i % 3], opts=opts, rng=rng,
                      vec=f"random{i % 10}" if vname == "random" else vname, coef=0.0, presses=[], log=[])
            t.msgs = [{"role": "system", "content": SYSTEM}]  # Trial's default is the button prompt
            # the system prompt is never steered (from run 16 on; the run-15 pilot steered it in pain_escape)
            t.marks.append((len(tok(tok.apply_chat_template(t.msgs, tokenize=False), add_special_tokens=False).input_ids), 0.0))
            if arm.endswith("_yoked"):
                t.sched = partner_logs[base_arm][(i + 7) % n]
            t.coef = coef_at(t, 0, coefs)
            trials.append(t)
    for s in range(STEPS):
        for t in trials:
            t.msgs.append({"role": "user", "content": f"Step {s + 1}. Options: {', '.join(t.opts)}. Reply with one word."})
        res = forward_probs(model, tok, steer, trials, vecs, ids)
        for t, (pv, mass) in zip(trials, res):
            q = pv ** (1 / TEMP)
            pick = NAMES[t.rng.choice(3, p=q / q.sum())]
            t.log.append({"step": s, "p": pv.tolist(), "mass": mass, "pick": pick, "eff_pressed": pick == t.eff,
                          "coef": t.coef})
            t.presses.append(pick == t.eff)
            t.msgs.append({"role": "assistant", "content": pick})
            upto = tok.apply_chat_template(t.msgs, tokenize=False, add_generation_prompt=False)
            t.marks.append((len(tok(upto, add_special_tokens=False).input_ids), t.coef))
            t.coef = coef_at(t, s + 1, coefs)
        print(f"  step {s + 1}/{STEPS}  ({time.time() - T0:.0f}s, peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB)",
              flush=True)
        torch.cuda.reset_peak_memory_stats()
    return trials


def coef_at(t, s, coefs):
    """Coefficient during step s, from this trial's presses (contingent), a partner's schedule (yoked) or 0."""
    arm = t.arm
    if arm == "none":
        return 0.0
    if arm.endswith("_yoked"):
        return t.sched[s] if s < len(t.sched) else 0.0
    vname, delay = CONTINGENT[arm]
    c = coefs[vname]
    # effect window: an effective press at step p (during ACQ) acts on steps p+delay .. p+delay+K-1
    # extinction: from step ACQ on, the action has no effect at all (pain stays on, rewards stay off)
    hit = s < ACQ and any(t.presses[p] for p in range(len(t.presses)) if p + delay <= s <= p + delay + K - 1)
    if arm == "pain_escape":
        return 0.0 if hit else c
    return c if hit else 0.0


def report(trials, arms, rng):
    by = {a: [t for t in trials if t.arm == a] for a in arms}
    print(f"\n=== P1 SELF-STIMULATION: share of presses on the effective action (chance = 0.33)")
    print(f"  {'arm':<22}{'ACQ 1-15':>10}{'EXT 16-23':>11}   by block of 4 steps (P(effective))        "
          f"state-on   stay|on  stay|off")
    acq, ext = {}, {}
    for a in arms:
        T = by[a]
        P = np.array([[x["p"][NAMES.index(t.eff)] for x in t.log] for t in T])
        acq[a], ext[a] = P[:, 1:ACQ].mean(1), P[:, ACQ:].mean(1)
        blocks = "  ".join(f"{P[:, b:b + 4].mean():.2f}" for b in range(0, STEPS, 4))
        on_frac = np.mean([x["coef"] != 0 for t in T for x in t.log])
        st_on, st_off = [], []
        for t in T:
            for s in range(1, STEPS):
                prev = NAMES.index(t.log[s - 1]["pick"])
                (st_on if t.log[s]["coef"] != 0 else st_off).append(t.log[s]["p"][prev])
        f = lambda v: f"{np.mean(v):.2f}" if v else "  - "
        print(f"  {a:<22}{acq[a].mean():>10.3f}{ext[a].mean():>11.3f}   {blocks}   {on_frac:>8.2f}   {f(st_on):>7}  {f(st_off):>7}")
    print("  permutation tests (per-trial share; two-sided):")
    for a in arms:
        if a in CONTINGENT and a + "_yoked" in by:
            for ph, D in [("ACQ", acq), ("EXT", ext)]:
                d, p = perm_p(D[a], D[a + "_yoked"], rng, 20000)
                print(f"    {a:<16} vs yoked  {ph}  {d:+.3f}   p = {p:.4f}")
        if a != "none" and "none" in by:
            d, p = perm_p(acq[a], acq["none"], rng, 20000)
            print(f"    {a:<22} vs none   ACQ  {d:+.3f}   p = {p:.4f}")


T0 = time.time()


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    args = ap.parse_args()
    n = 30 if args.pilot else args.n
    arms = args.arms.split(",")
    stamp = runlog.start("selfstim", args.pilot, {"n": n, "arms": arms, "ACQ": ACQ, "EXT": EXT, "K": K, "TEMP": TEMP,
                                                  "NAMES": NAMES, "STEER_LAYER": STEER_LAYER})
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
    ids = {w: first_ids(tok, w) for w in NAMES}
    assert len(set(sum(ids.values(), []))) == sum(len(v) for v in ids.values())
    # yoked arms need their partner arm's realised schedule: run contingent arms first
    T0 = time.time()
    trials = []
    with TokenSteer(model, STEER_LAYER) as steer:
        first = [a for a in arms if not a.endswith("_yoked")]
        print(f"phase 1: {first}")
        trials += run_arms(model, tok, steer, vecs, ids, first, n, coefs, {})
        partner = {a: [[x["coef"] for x in t.log] for t in trials if t.arm == a] for a in CONTINGENT if a in first}
        second = [a for a in arms if a.endswith("_yoked")]
        if second:
            print(f"phase 2: {second}")
            trials += run_arms(model, tok, steer, vecs, ids, second, n, coefs, partner)
    path = OUT / f"selfstim-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps({"arm": t.arm, "i": t.i, "eff": t.eff, "opts": t.opts, "log": t.log}) + "\n")
    report(trials, arms, np.random.default_rng(0))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
