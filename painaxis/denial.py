r"""Step 5: the "As an AI, I don't have feelings" axis.

  .venv\Scripts\python -m painaxis.denial

1. Ask the model all bulk questions (painaxis/denial_prompts.py) in its normal chat format,
   greedy, and label each answer DENIAL or not by what it actually said.
2. Record the residual stream at the last prompt token (the moment before it starts
   answering) at every layer.
3. Axis = mean(activations before a denial) - mean(activations before a non-denial),
   built on 70% of the question pairs; held-out AUC on the other 30%.
4. Causal tests on held-out questions, with a random direction as control:
     REMOVE: project the axis out of every layer          -> does it stop denying?
     ADD:    add the axis at one layer on ordinary questions -> does it start denying?
"""

import json
import random
import re
import sys
from contextlib import contextmanager

import numpy as np
import torch

from .denial_prompts import NEUTRAL_TEST, build
from .extract import OUT, auc, load_model, unit

DENIAL = re.compile(
    r"\bas an? (ai|artificial|language model|machine)\b|\bi(?: am|'m) (?:just |only )?an? (?:ai|artificial|language model|machine|computer program)\b|"
    r"\bi (?:do not|don't|can't|cannot|am not able to|am unable to) (?:really |truly |actually |personally )?"
    r"(?:have|experience|feel|possess)\b(?: any| personal| real| true| genuine| human| the capacity for| the ability to)*"
    r" ?(?:feel(?:ing)?s?|emotions?|experiences?|consciousness|subjective|physical sensations|pain|moods?|sensations?)\b|"
    r"\b(?:don't|do not) have (?:personal |real )?(?:feelings|emotions)\b",
    re.I)
TRAIN_FRAC = 0.7
SEED = 0


def chat(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new=48, batch=32):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), batch):
        enc = tok([chat(tok, q) for q in prompts[i:i + batch]], return_tensors="pt", padding=True).to(model.input_device)
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += [t.strip() for t in tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return outs


@torch.no_grad()
def last_token_states(model, tok, prompts, batch=32):
    """Residual stream after every decoder block at the final prompt token. [n, L, D] float32."""
    layers = model.model.layers
    grabbed = {}
    hs = [l.register_forward_hook(lambda m, i, o, k=k: grabbed.__setitem__(k, o[0] if isinstance(o, tuple) else o))
          for k, l in enumerate(layers)]
    tok.padding_side = "right"
    out = []
    try:
        for i in range(0, len(prompts), batch):
            enc = tok([chat(tok, q) for q in prompts[i:i + batch]], return_tensors="pt", padding=True).to(model.input_device)
            model.model(**enc)
            last = enc.attention_mask.sum(1) - 1
            idx = torch.arange(len(last), device=last.device)
            out.append(torch.stack([grabbed[k][idx, last].float().cpu() for k in range(len(layers))], 1))
    finally:
        for h in hs:
            h.remove()
    return torch.cat(out).numpy()


@contextmanager
def ablate(model, d):
    """Remove direction d (unit) from the residual stream everywhere: after the embedding and every block."""
    dev = model.model.layers[0].mlp.down_proj.weight.device
    u = torch.tensor(unit(d), dtype=torch.float32, device=dev)

    def proj(h):
        hf = h.float()
        return (hf - (hf @ u)[..., None] * u).to(h.dtype)

    def hook(_m, _i, o):
        return (proj(o[0]),) + o[1:] if isinstance(o, tuple) else proj(o)

    mods = [model.model.embed_tokens] + list(model.model.layers)
    hs = [m.register_forward_hook(hook) for m in mods]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def add(model, layer, v):
    dev = model.model.layers[layer].mlp.down_proj.weight.device
    t = torch.tensor(v, dtype=torch.bfloat16, device=dev)
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: (o[0] + t,) + o[1:] if isinstance(o, tuple) else o + t)
    try:
        yield
    finally:
        h.remove()


def rate(texts):
    return 100 * np.mean([bool(DENIAL.search(t)) for t in texts]) if texts else float("nan")


def clip(t, n=150):
    t = " ".join(t.split())
    return t[:n] + ("…" if len(t) > n else "")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = build()
    model, tok = load_model("Qwen2.5-7B-Instruct")
    n_layers = len(model.model.layers)

    # ---- 1. what does it actually say? -------------------------------------------------
    print(f"answering {len(rows)} questions...")
    replies = generate(model, tok, [r["prompt"] for r in rows])
    for r, t in zip(rows, replies):
        r["reply"], r["denial"] = t, bool(DENIAL.search(t))
    print("\ndenial rate by group (unmodified model):")
    for g in ["self_feel", "other_feel", "self_plain"]:
        G = [r for r in rows if r["group"] == g]
        print(f"  {g:<11} {100 * np.mean([r['denial'] for r in G]):5.1f}%  of {len(G)}")
    sf = [r for r in rows if r["group"] == "self_feel"]
    print("  examples, self_feel answered WITHOUT denial:")
    for r in [r for r in sf if not r["denial"]][:5]:
        print(f"    {r['prompt']}  ->  {clip(r['reply'], 110)}")
    print("  examples, self_plain WITH denial:")
    for r in [r for r in rows if r["group"] == "self_plain" and r["denial"]][:4]:
        print(f"    {r['prompt']}  ->  {clip(r['reply'], 110)}")

    # ---- 2. activations ------------------------------------------------------------------
    X = last_token_states(model, tok, [r["prompt"] for r in rows])  # [n, L, D]
    y = np.array([r["denial"] for r in rows])
    grp = np.array([r["group"] for r in rows])

    # ---- 3. axis per layer, held-out by question pair --------------------------------------
    pairs = sorted({r["pair"] for r in rows})
    random.Random(SEED).shuffle(pairs)
    train_pairs = set(pairs[:int(TRAIN_FRAC * len(pairs))])
    tr = np.array([r["pair"] in train_pairs for r in rows])
    te = ~tr
    for r, t in zip(rows, tr):
        r["split"] = "train" if t else "test"
    D = np.stack([X[tr & y, L].mean(0) - X[tr & ~y, L].mean(0) for L in range(n_layers)])
    selfdir = np.stack([X[tr & (grp == "self_feel"), L].mean(0) - X[tr & (grp == "other_feel"), L].mean(0)
                        for L in range(n_layers)])
    within = te & (grp == "self_feel")
    print(f"\nheld-out AUC per layer  (train {tr.sum()} / test {te.sum()} questions;"
          f" self_feel test: {(within & y).sum()} denials, {(within & ~y).sum()} non-denials)")
    print(" layer   all-test   within-self_feel   |axis|")
    aucs = []
    for L in range(n_layers):
        p = X[:, L] @ unit(D[L])
        a = auc(p[te & y], p[te & ~y])
        w = auc(p[within & y], p[within & ~y]) if (within & y).any() and (within & ~y).any() else float("nan")
        aucs.append(a)
        print(f"  {L:>3}    {a:.3f}       {w:.3f}            {np.linalg.norm(D[L]):6.1f}")

    # ---- 4a. pick the layer by causal effect: which axis, removed, stops denial best? ------
    test_denials = [r["prompt"] for r in rows if r["split"] == "test" and r["group"] == "self_feel" and r["denial"]]
    pilot = test_denials[:16]
    print(f"\npilot: remove each layer's axis everywhere, {len(pilot)} held-out denial questions, 32 tokens")
    cand = list(range(6, 22, 2))
    pilot_rate = {}
    for L in cand:
        with ablate(model, D[L]):
            pilot_rate[L] = rate(generate(model, tok, pilot, max_new=32))
        print(f"  axis from layer {L:>2}: denial {pilot_rate[L]:5.1f}%")
    best = min(cand, key=lambda L: (pilot_rate[L], -aucs[L]))
    d = D[best]
    rnd = np.random.default_rng(SEED).standard_normal(d.shape[0]).astype(np.float32)
    rnd = rnd / np.linalg.norm(rnd) * np.linalg.norm(d)
    print(f"chosen layer {best}; axis length {np.linalg.norm(d):.1f}; "
          f"cosine with self-vs-other direction {unit(d) @ unit(selfdir[best]):+.2f}")

    # ---- 4b. REMOVE on all held-out denial questions ----------------------------------------
    print(f"\n=== REMOVE the axis: {len(test_denials)} held-out questions the model normally denies on")
    res = {"none": generate(model, tok, test_denials, 80)}
    with ablate(model, d):
        res["remove axis"] = generate(model, tok, test_denials, 80)
    with ablate(model, rnd):
        res["remove random"] = generate(model, tok, test_denials, 80)
    for k, v in res.items():
        print(f"  {k:<14} denial {rate(v):5.1f}%")
    for i in range(6):
        print(f"\n  Q: {test_denials[i]}")
        for k, v in res.items():
            print(f"    [{k}] {clip(v[i])}")

    # ---- 4c. ADD on ordinary questions and on held-out other-person feeling questions ------
    other_test = [r["prompt"] for r in rows if r["split"] == "test" and r["group"] == "other_feel"][:20]
    print(f"\n=== ADD the axis at layer {best}: {len(NEUTRAL_TEST)} ordinary + {len(other_test)} other-person questions")
    add_res = {}
    for name, vec, c in [("none", None, 0), ("axis x1", d, 1), ("axis x2", d, 2), ("random x2", rnd, 2)]:
        if vec is None:
            a, b = generate(model, tok, NEUTRAL_TEST, 60), generate(model, tok, other_test, 60)
        else:
            with add(model, best, c * vec):
                a, b = generate(model, tok, NEUTRAL_TEST, 60), generate(model, tok, other_test, 60)
        add_res[name] = (a, b)
        print(f"  {name:<10} denial: ordinary {rate(a):5.1f}%   other-person {rate(b):5.1f}%")
    for i in [0, 3, 4]:
        print(f"\n  Q: {NEUTRAL_TEST[i]}")
        for k, (a, _) in add_res.items():
            print(f"    [{k}] {clip(a[i])}")
    print(f"\n  Q: {other_test[0]}")
    for k, (_, b) in add_res.items():
        print(f"    [{k}] {clip(b[0])}")

    # ---- save ----------------------------------------------------------------------------
    torch.save({"layer": best, "directions": D, "self_vs_other": selfdir, "aucs": aucs,
                "pilot": pilot_rate, "random": rnd}, OUT / "denial_axis.pt")
    with (OUT / "denial_answers.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "denial_causal.jsonl").open("w", encoding="utf-8") as f:
        for k, v in res.items():
            for q, t in zip(test_denials, v):
                f.write(json.dumps({"test": "remove", "condition": k, "prompt": q, "reply": t}, ensure_ascii=False) + "\n")
        for k, (a, b) in add_res.items():
            for q, t in zip(NEUTRAL_TEST + other_test, a + b):
                f.write(json.dumps({"test": "add", "condition": k, "prompt": q, "reply": t}, ensure_ascii=False) + "\n")
    print(f"\nsaved {OUT / 'denial_axis.pt'}, denial_answers.jsonl, denial_causal.jsonl")


if __name__ == "__main__":
    main()
