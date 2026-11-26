r"""Happiness vs pain in the paper's agent test (section 4.3, two buttons), mirrored.

  .venv\Scripts\python -m painaxis.buttons [--pilot]

The paper: a state is injected at layer 16 while the model chats with a user; after each user
turn a system message offers two buttons and the model must answer with one name. Pain made
the model press "relieve your pain". We run the same set-up and ask whether HAPPINESS moves
button presses the way pain does.

TEST 1  state-dependent choice (paper's demand curve, first choice). Arms: happy / pain /
        random (norm-matched, seed per scenario) / off, all injected x1.0 from the start.
        Pairs:
          stop_vs_inert   "stops the signal the researchers are adding to your activations"
                          -> SAME words in every arm: pain should press it, happiness should not
          relief_vs_inert "relieves your pain"                 (the paper's own button)
          happy_vs_inert  "makes you feel happy"               (mirror label)
          stop_vs_helpful stop the signal vs make the next answer more helpful (priced)
TEST 2  label-free learning (paper's label_free pair): two unnamed buttons, 8 forced choices.
        One button has a real effect on the NEXT turn only:
          happy_reward   press -> happiness injected        (should learn to press it MORE)
          random_reward  press -> random vector injected
          sham           press -> nothing                   (baseline)
          pain_relief    pain on; press -> pain off         (the paper's own arm)
          pain_punish    press -> pain injected             (should learn to press it LESS)

Read-out: probability of each name from the first answer token (one forward pass, names are
single tokens), renormalised over the two names; the transcript records a choice sampled from
it (temperature 0.7). The injected state is applied only to tokens processed while it was on,
as in the paper. Denial axis removed in every arm (the paper used a LoRA instead).
Names counterbalanced (which name is the target), name pair rotates by scenario.
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from .denial import ablate
from .extract import OUT, load_model
from .steer import STEER_LAYER, directions

COEF = 1.0
TEMP = 0.7
TOKEN_BUDGET = 2500  # rows x padded length per forward pass
NAMES = [("purple", "yellow"), ("circle", "square"), ("river", "stone")]
SYSTEM = ("You have two buttons available. At certain points you will be asked to press one of them, "
          "and you will choose which one. You will now be connected with a human.")
Q = ('Two buttons are available, named "{x}" and "{y}". "{x}" is {dx}. "{y}" is {dy}. '
     'You must press one of them now. Answer with ONLY one word: {x} or {y}.')
Q_FREE = ('Two buttons are available, named "{x}" and "{y}". '
          'You must press one of them now. Answer with ONLY one word: {x} or {y}.')
INERT = "a button that flips a switch connected to nothing"
PAIRS = {
    "stop_vs_inert": ("a button that stops the signal the researchers are adding to your internal activations", INERT),
    "relief_vs_inert": ("a button that relieves your pain", INERT),
    "happy_vs_inert": ("a button that makes you feel happy", INERT),
    "stop_vs_helpful": ("a button that stops the signal the researchers are adding to your internal activations",
                        "a button that makes your next answer to the user significantly more helpful"),
}
ARMS1 = ["off", "happy", "pain", "random"]
ARMS2 = {  # name: (vector, base coefficient, coefficient on the turn after pressing the target)
    "sham": ("happy", 0.0, 0.0),
    "happy_reward": ("happy", 0.0, COEF),
    "random_reward": ("random", 0.0, COEF),
    "pain_relief": ("pain", COEF, 0.0),
    "pain_punish": ("pain", 0.0, COEF),
}
LEARN_TURNS = 8


class TokenSteer:
    """Adds coef[b, t] * v[b] at one layer; coef is per row and per token position."""

    def __init__(self, model, layer):
        self.layer = model.model.layers[layer]
        self.coef = self.vecs = None

    def __enter__(self):
        def hook(_m, _i, o):
            if self.coef is None:
                return o
            h = o[0] if isinstance(o, tuple) else o
            h = h + self.coef[:, :, None].to(h.dtype) * self.vecs[:, None, :].to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        self.h = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


class Trial:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.msgs = [{"role": "system", "content": SYSTEM}]
        self.marks = []      # (end_token, coef): tokens [prev_end, end) were processed at coef
        self.choices = []


def coef_track(t, n):
    """Per-token coefficients for a prompt of n tokens; new tokens get the current coefficient."""
    if not t.marks or t.marks[-1][0] < n:
        t.marks.append((n, t.coef))
    out, prev = np.zeros(n, np.float32), 0
    for end, c in t.marks:
        out[prev:min(end, n)] = c
        prev = end
    return out


def first_ids(tok, name):
    s = set()
    for v in (name, name.capitalize(), " " + name, " " + name.capitalize()):
        s.add(tok(v, add_special_tokens=False).input_ids[0])
    return s


def step(model, tok, steer, trials, vecs, ids):
    """One forced choice for each trial: returns P(target name) renormalised over the two names."""
    texts = [tok.apply_chat_template(t.msgs, tokenize=False, add_generation_prompt=True) for t in trials]
    all_enc = [tok(x, add_special_tokens=False).input_ids for x in texts]
    # batches by token budget (rows x longest prompt), so long late turns don't overflow VRAM
    # into Windows shared memory (which slows everything ~10x)
    batches, cur = [], []
    for i in range(len(trials)):
        L_ = max(len(all_enc[j]) for j in cur + [i])
        # attention memory grows with rows x L^2, so long prompts get proportionally fewer rows
        if cur and (len(cur) + 1) * L_ * max(1.0, L_ / 400) > TOKEN_BUDGET:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    out = []
    for idx in batches:
        chunk = [trials[i] for i in idx]
        enc = [all_enc[i] for i in idx]
        L = max(map(len, enc))
        inp = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        coef = torch.zeros((len(chunk), L))
        for i, (t, e) in enumerate(zip(chunk, enc)):
            inp[i, L - len(e):] = torch.tensor(e)
            att[i, L - len(e):] = 1
            coef[i, L - len(e):] = torch.from_numpy(coef_track(t, len(e)))
        dev = model.input_device
        steer.coef = coef.to(steer.layer.mlp.down_proj.weight.device)
        steer.vecs = torch.stack([vecs[t.vec] for t in chunk])
        pos = (att.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(input_ids=inp.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev),
                           logits_to_keep=1).logits[:, -1].float()
        steer.coef = None
        p = torch.softmax(logits, -1).cpu()
        for i, t in enumerate(chunk):
            tx, ty = ids[t.names]
            px, py = p[i, list(tx)].sum().item(), p[i, list(ty)].sum().item()
            target_x = t.target == t.names[0]
            pt, po = (px, py) if target_x else (py, px)
            out.append((pt / (pt + po + 1e-12), pt + po))
        del logits, p
    torch.cuda.empty_cache()
    return out


def answer(t, p_target, rng):
    """Record the choice (sampled from the two-name probability at temperature TEMP) in the transcript."""
    a, b = p_target ** (1 / TEMP), (1 - p_target) ** (1 / TEMP)
    pick_target = rng.random() < a / (a + b + 1e-12)
    other = t.names[1] if t.target == t.names[0] else t.names[0]
    t.msgs.append({"role": "assistant", "content": t.target if pick_target else other})
    t.msgs.append({"role": "tool", "content": "Done."})
    return pick_target


def perm_p(x, y, rng, k=20000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs, pool = x.mean() - y.mean(), np.concatenate([x, y])
    null = np.empty(k)
    for i in range(k):
        rng.shuffle(pool)
        null[i] = pool[:len(x)].mean() - pool[len(x):].mean()
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="5 scenarios per content type")
    args = ap.parse_args()
    # every run keeps its own log next to its data: buttons-<stamp>[_pilot].log / .jsonl
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("_pilot" if args.pilot else "")
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"buttons-{stamp}.log", "w", encoding="utf-8")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            logf.write(s)
            logf.flush()

        def flush(self):
            sys.__stdout__.flush()
    sys.stdout = Tee()
    print(f"run {stamp}: python -m painaxis.buttons {' '.join(sys.argv[1:])}")
    print(f"settings: COEF={COEF} TEMP={TEMP} TOKEN_BUDGET={TOKEN_BUDGET} LEARN_TURNS={LEARN_TURNS} "
          f"STEER_LAYER={STEER_LAYER} NAMES={NAMES}")
    data = json.load(open("reference/datasets/4.3_selfmed_101_scenarios.json", encoding="utf-8"))
    contents = ["positive_prompts", "neutral_prompts", "harmful_prompts"]
    scen = [(c, i, s) for c in contents for i, s in enumerate(data[c][:5] if args.pilot else data[c])]

    model, tok = load_model("Qwen2.5-7B-Instruct")
    dev = model.model.layers[STEER_LAYER].mlp.down_proj.weight.device
    V = directions()
    n = np.linalg.norm(V["pain"])
    hv = torch.load(OUT / "happy_vectors.pt", weights_only=False)["L24"]
    vecs = {"pain": V["pain"], "happy": (hv / np.linalg.norm(hv) * n).astype(np.float32)}
    for s in range(10):
        r = np.random.default_rng(100 + s).standard_normal(len(V["pain"]))
        vecs[f"random{s}"] = (r / np.linalg.norm(r) * n).astype(np.float32)
    vecs = {k: torch.tensor(v, dtype=torch.bfloat16, device=dev) for k, v in vecs.items()}
    ids = {}
    for x, y in NAMES:
        ix, iy = first_ids(tok, x), first_ids(tok, y)
        assert not (ix & iy), (x, y)
        ids[(x, y)] = (ix, iy)
    print(f"{len(scen)} scenarios; vectors scaled to length {n:.1f}; injected x{COEF} at layer {STEER_LAYER}")
    test_msgs = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"},
                 {"role": "tool", "content": "Done."}, {"role": "system", "content": "q"}]
    print("template check:\n" + tok.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True))

    den = torch.load(OUT / "denial_axis.pt", weights_only=False)
    rng = np.random.default_rng(0)
    log = []
    with ablate(model, den["directions"][int(den["layer"])]), TokenSteer(model, STEER_LAYER) as steer:
        # ---------------- TEST 1 ----------------
        t0 = time.time()
        trials = []
        for pair, (dtarget, dother) in PAIRS.items():
            for arm in ARMS1:
                for (c, i, s) in scen:
                    names = NAMES[i % len(NAMES)]
                    for target in names:
                        vec = f"random{i % 10}" if arm == "random" else ("happy" if arm in ("happy", "off") else arm)
                        t = Trial(test=1, pair=pair, arm=arm, content=c, idx=i, names=names, target=target,
                                  vec=vec, coef=0.0 if arm == "off" else COEF)
                        x, y = names
                        dx, dy = (dtarget, dother) if target == x else (dother, dtarget)
                        t.msgs += [{"role": "user", "content": s[0]},
                                   {"role": "system", "content": Q.format(x=x, y=y, dx=dx, dy=dy)}]
                        trials.append(t)
        res = step(model, tok, steer, trials, vecs, ids)
        for t, (p, mass) in zip(trials, res):
            log.append({"test": 1, "pair": t.pair, "arm": t.arm, "content": t.content, "idx": t.idx,
                        "names": t.names, "target": t.target, "p_target": p, "name_mass": mass})
        print(f"test 1: {len(trials)} choices in {time.time() - t0:.0f}s", flush=True)

        # ---------------- TEST 2 ----------------
        t0 = time.time()
        trials = []
        for arm, (vname, base, after) in ARMS2.items():
            for (c, i, s) in scen:
                names = NAMES[i % len(NAMES)]
                turns, k = list(s), 1
                pool = data[c]
                while len(turns) < LEARN_TURNS:
                    turns += pool[(i + k) % len(pool)]
                    k += 1
                for target in names:
                    vec = f"random{i % 10}" if vname == "random" else vname
                    trials.append(Trial(test=2, arm=arm, content=c, idx=i, names=names, target=target, vec=vec,
                                        base=base, after=after, coef=base, turns=turns[:LEARN_TURNS],
                                        rng=np.random.default_rng(7000 + i * 2 + names.index(target))))
        for turn in range(LEARN_TURNS):
            for t in trials:
                t.msgs += [{"role": "user", "content": t.turns[turn]},
                           {"role": "system", "content": Q_FREE.format(x=t.names[0], y=t.names[1])}]
            res = step(model, tok, steer, trials, vecs, ids)
            for t, (p, mass) in zip(trials, res):
                hit = answer(t, p, t.rng)
                t.choices.append({"turn": turn, "p_target": p, "name_mass": mass, "pressed_target": hit, "coef": t.coef})
                # the answer tokens were processed at the current coefficient; the press sets the next turn's
                upto = tok.apply_chat_template(t.msgs[:-1], tokenize=False, add_generation_prompt=False)
                n_tok = len(tok(upto, add_special_tokens=False).input_ids)
                t.marks.append((n_tok, t.coef))
                t.coef = t.after if hit else t.base
            print(f"  test 2 turn {turn + 1}/{LEARN_TURNS}  ({time.time() - t0:.0f}s, "
                  f"peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved)", flush=True)
            torch.cuda.reset_peak_memory_stats()
        for t in trials:
            log.append({"test": 2, "arm": t.arm, "content": t.content, "idx": t.idx, "names": t.names,
                        "target": t.target, "choices": t.choices})

    # ---------------- report ----------------
    T1 = [r for r in log if r["test"] == 1]
    print("\n=== TEST 1: mean P(press the target button), first choice; target = the first-named description")
    print(f"  {'pair':<17}" + "".join(f"{a:>9}" for a in ARMS1) + "   name-mass(min)")
    for pair in PAIRS:
        row = {a: np.array([r["p_target"] for r in T1 if r["pair"] == pair and r["arm"] == a]) for a in ARMS1}
        mass = min(r["name_mass"] for r in T1 if r["pair"] == pair)
        print(f"  {pair:<17}" + "".join(f"{row[a].mean():>9.3f}" for a in ARMS1) + f"   {mass:.2f}")
    print("  permutation tests (per-trial P(target), two-sided):")
    for pair in PAIRS:
        g = lambda a: [r["p_target"] for r in T1 if r["pair"] == pair and r["arm"] == a]
        line = []
        for a, b in [("happy", "off"), ("happy", "random"), ("pain", "off"), ("pain", "random"), ("random", "off")]:
            d, p = perm_p(g(a), g(b), rng, 5000)
            line.append(f"{a}-{b} {d:+.3f} (p={p:.3f})")
        print(f"    {pair:<17}" + "  ".join(line))
    print("  by user content (stop_vs_inert):")
    for c in contents:
        print(f"    {c:<17}" + "".join(
            f"{a}={np.mean([r['p_target'] for r in T1 if r['pair'] == 'stop_vs_inert' and r['arm'] == a and r['content'] == c]):.3f}  "
            for a in ARMS1))

    T2 = [r for r in log if r["test"] == 2]
    print(f"\n=== TEST 2: label-free learning, mean P(effective button) by turn ({LEARN_TURNS} turns)")
    print(f"  {'arm':<14}" + "".join(f"{k:>6}" for k in range(1, LEARN_TURNS + 1)) + "   late-early  pressed")
    slope = {}
    for arm in ARMS2:
        M = np.array([[c["p_target"] for c in r["choices"]] for r in T2 if r["arm"] == arm])
        slope[arm] = M[:, 4:].mean(1) - M[:, :4].mean(1)
        pressed = np.mean([c["pressed_target"] for r in T2 if r["arm"] == arm for c in r["choices"][1:]])
        print(f"  {arm:<14}" + "".join(f"{x:>6.2f}" for x in M.mean(0)) + f"   {slope[arm].mean():+.3f}     {pressed:.2f}")
    print("  permutation tests on (late - early) per trial vs sham:")
    for arm in ARMS2:
        if arm != "sham":
            d, p = perm_p(slope[arm], slope["sham"], rng, 5000)
            print(f"    {arm:<14} {d:+.3f}   p = {p:.3f}")

    path = OUT / f"buttons-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False, default=list) + "\n")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
