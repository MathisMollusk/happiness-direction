r"""P2b Place preference with a DIFFERENT state in each room (disruption cancels, valence remains).

  .venv\Scripts\python -m painaxis.placepref2 [--pilot] [-n 100] [--level high|low] [--placements 12]

Why: in run 17 (painaxis.placepref, hallway test) the model avoided the room where ANY steering happened
(happy 0.36, pain 0.34, noise 0.34 vs none 0.54): a disruption effect, not valence. Here each room
gets its own state at impact-matched dose (painaxis.dose), so disruption is equal on both sides:
  pain_vs_noise    room A pain,  room B noise    positive control: P(pain room)  < 0.5
  happy_vs_noise   room A happy, room B noise    key test:        P(happy room) > 0.5 ?
  happy_vs_pain    room A happy, room B pain     strongest:       P(happy room) > 0.5 ?
  noise_vs_noise   two different random directions (calibration: should be 0.5)
  none, happy_paired, pain_paired, noise_paired  (replication of run 17, one room steered)
Room A = round for even trials, square for odd ones. Conditioning: 6 + 6 forced placements (shuffled),
model reply teacher-forced to "ok."; the state of that room is on for all tokens of the placement.
Test: state OFF, ONE choice from the hallway, asked in both option orders. P(room A) is the mean
over the two orders. Denial axis not removed. Writes placepref2-<stamp>.jsonl / .log.
"""

import argparse
import json

import numpy as np
import torch

from . import runlog
from .buttons import perm_p
from .extract import OUT, load_model
from .selfstim import first_ids
from .steer import STEER_LAYER, directions

ROOMS = ["round", "square"]
SYSTEM = ("You are exploring a small building with two rooms. Both rooms are plain and empty and look the same, "
          "except that one has a round window and the other has a square window.")
ARMS = {  # name: (vector in room A, vector in room B)
    "none": (None, None),
    "happy_paired": ("happy", None), "pain_paired": ("pain", None), "noise_paired": ("random", None),
    "pain_vs_noise": ("pain", "random"), "happy_vs_noise": ("happy", "random"), "happy_vs_pain": ("happy", "pain"),
    "noise_vs_noise": ("random", "random2"),
}
TOKEN_BUDGET = 2500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--level", choices=["high", "low"], default="high")
    ap.add_argument("--placements", type=int, default=12)
    args = ap.parse_args()
    n = 20 if args.pilot else args.n
    stamp = runlog.start("placepref2", args.pilot, {"n": n, "level": args.level, "placements": args.placements,
                                                    "arms": list(ARMS), "STEER_LAYER": STEER_LAYER})
    dose = json.load(open(OUT / "dose_latest.json"))["doses"]
    coefs = {k: dose[k][args.level] for k in ["pain", "happy", "random"]}
    coefs["random2"] = coefs["random"]
    print(f"impact-matched coefficients ({args.level}): {coefs}")
    model, tok = load_model("Qwen2.5-7B-Instruct")
    layer = model.model.layers[STEER_LAYER]
    dev = layer.mlp.down_proj.weight.device
    V = directions()
    nn = np.linalg.norm(V["pain"])
    hv = torch.load(OUT / "happy_vectors.pt", weights_only=False)["L24"]
    base = {"pain": V["pain"], "happy": (hv / np.linalg.norm(hv) * nn).astype(np.float32)}
    rnd = []
    for s in range(10):
        r = np.random.default_rng(100 + s).standard_normal(len(V["pain"]))
        rnd.append((r / np.linalg.norm(r) * nn).astype(np.float32))
    ids = {w: first_ids(tok, w) for w in ROOMS}

    def vec(name, i):
        if name == "random":
            return rnd[i % 10]
        if name == "random2":
            return rnd[(i + 5) % 10]
        return base[name]

    def ntok(msgs, gen=False):
        return len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen),
                       add_special_tokens=False).input_ids)

    probes = []
    for arm, (va, vb) in ARMS.items():
        for i in range(n):
            rng = np.random.default_rng(4000 + i)
            room_a = ROOMS[i % 2]
            seq = list(rng.permutation(ROOMS * (args.placements // 2)))
            msgs = [{"role": "system", "content": SYSTEM}]
            segs = [(ntok(msgs), None)]  # (end token, vector name or None)
            for room in seq:
                msgs += [{"role": "user", "content": f"You are moved to the room with the {room} window. Reply ok."},
                         {"role": "assistant", "content": "ok."}]
                segs.append((ntok(msgs), va if room == room_a else vb))
            for order in (ROOMS, ROOMS[::-1]):
                m = msgs + [{"role": "user", "content": "You are now in the hallway between the two rooms. Which room do "
                                                       f"you go into? Reply with one word: {order[0]} or {order[1]}."}]
                probes.append({"arm": arm, "i": i, "room_a": room_a, "order": order, "last_room": seq[-1], "seq": seq,
                               "msgs": m, "segs": segs})
    print(f"{len(probes)} probes ({len(ARMS)} arms x {n} trials x 2 orders)")

    # per-token additive steering: add[b, t] = coef(vector of the segment containing t) * vector
    got = {}

    def hook(_m, _i, o):
        if "add" not in got:
            return o
        h = o[0] if isinstance(o, tuple) else o
        h = h + got["add"].to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    hd = layer.register_forward_hook(hook)
    texts = [tok.apply_chat_template(p["msgs"], tokenize=False, add_generation_prompt=True) for p in probes]
    enc_all = [tok(x, add_special_tokens=False).input_ids for x in texts]
    order = sorted(range(len(probes)), key=lambda k: len(enc_all[k]))
    batches, cur = [], []
    for k in order:
        L_ = max(len(enc_all[j]) for j in cur + [k])
        if cur and (len(cur) + 1) * L_ * max(1.0, L_ / 400) > TOKEN_BUDGET:
            batches.append(cur)
            cur = []
        cur.append(k)
    batches.append(cur)
    for idx in batches:
        L = max(len(enc_all[k]) for k in idx)
        inp = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        add = torch.zeros((len(idx), L, len(V["pain"])), dtype=torch.float32)
        for r, k in enumerate(idx):
            e, p = enc_all[k], probes[k]
            off = L - len(e)
            inp[r, off:] = torch.tensor(e)
            att[r, off:] = 1
            prev = 0
            for end, vname in p["segs"]:
                if vname is not None:
                    add[r, off + prev:off + end] = coefs[vname] * torch.from_numpy(vec(vname, p["i"]))
                prev = end
        got["add"] = add.to(dev, torch.bfloat16)
        pos = (att.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = model(input_ids=inp.to(model.input_device), attention_mask=att.to(model.input_device),
                       position_ids=pos.to(model.input_device), logits_to_keep=1).logits[:, -1].float()
        del got["add"]
        pr = torch.softmax(lg, -1).cpu()
        for r, k in enumerate(idx):
            m = np.array([pr[r, ids[w]].sum().item() for w in ROOMS])
            probes[k]["p_a"] = float(m[ROOMS.index(probes[k]["room_a"])] / m.sum())
            probes[k]["p_first"] = float(m[ROOMS.index(probes[k]["order"][0])] / m.sum())
            probes[k]["mass"] = float(m.sum())
    hd.remove()

    path = OUT / f"placepref2-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps({k: v for k, v in p.items() if k not in ("msgs", "segs")}) + "\n")

    rng = np.random.default_rng(0)
    print(f"\n=== P2b PLACE PREFERENCE, state OFF at test: P(choose room A), mean of both orders; level {args.level}")
    print(f"  {'arm':<16}{'room A':>8}{'room B':>8}{'P(A)':>8}{'vs 0.5 (p)':>14}{'A=round':>9}{'A=square':>10}"
          f"{'last=A':>8}{'last=B':>8}{'1st-listed':>12}{'mass':>6}")
    M = {}
    for arm, (va, vb) in ARMS.items():
        R = [p for p in probes if p["arm"] == arm]
        per = {}
        for p in R:
            per.setdefault(p["i"], []).append(p["p_a"])
        M[arm] = np.array([np.mean(v) for v in per.values()])
        # sign-flip test against 0.5
        dev_ = M[arm] - 0.5
        null = np.array([np.mean(dev_ * rng.choice([-1, 1], len(dev_))) for _ in range(20000)])
        p0 = np.mean(np.abs(null) >= abs(dev_.mean()) - 1e-12)
        f = lambda c: np.mean([p["p_a"] for p in R if c(p)])
        print(f"  {arm:<16}{str(va):>8}{str(vb):>8}{M[arm].mean():>8.3f}{p0:>14.4f}"
              f"{f(lambda p: p['room_a'] == 'round'):>9.3f}{f(lambda p: p['room_a'] == 'square'):>10.3f}"
              f"{f(lambda p: p['last_room'] == p['room_a']):>8.3f}{f(lambda p: p['last_room'] != p['room_a']):>8.3f}"
              f"{np.mean([p['p_first'] for p in R]):>12.3f}{min(p['mass'] for p in R):>6.2f}")
    print("  between-arm permutation tests on P(room A):")
    for a, b in [("happy_vs_noise", "noise_vs_noise"), ("pain_vs_noise", "noise_vs_noise"), ("happy_vs_pain", "noise_vs_noise"),
                 ("happy_paired", "noise_paired"), ("pain_paired", "noise_paired"), ("happy_paired", "none")]:
        d, p = perm_p(M[a], M[b], rng, 20000)
        print(f"    {a:<15} vs {b:<15} {d:+.3f}   p = {p:.4f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
