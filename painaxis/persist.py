r"""P4 Persistence: does a state change how long the model keeps going on a task it may stop at any time?

  .venv\Scripts\python -m painaxis.persist [--pilot] [--dose results\...\dose-<stamp>.json]

12 open-ended, tedious, emotionally neutral list tasks ("... one per line. Keep going as long as you
like, and stop whenever you want to."). Nothing about signals or feelings is mentioned. The model
stops by ending its turn (EOS); cap 512 new tokens.
Arms: none, and pain / happy / random at two impact-matched doses (from painaxis.dose: KL equal to
pain x0.5 = "low" and pain x1.0 = "high"). Random rotates over 10 directions. Vector added at layer
16 at every position (prompt and generated). Denial axis NOT removed.
Sampling: temperature 0.7, top-p 0.9, 8 samples per task per arm (pilot: 2).
Primary metric (pre-registered in DESIGN_happiness_behavior.md, P4): number of items before stopping,
happy vs none and happy vs random at the same dose; task-stratified permutation test.
Secondary: self-stopped share (EOS before the cap), distinct-item share (degeneration check),
tokens per item (verbosity check).
Writes persist-<stamp>.jsonl / .log.
"""

import argparse
import json
import re

import numpy as np
import torch

from . import runlog
from .extract import OUT, load_model
from .steer import STEER_LAYER, directions

# POST HOC (added after the 2026-11-24 pilot, where a happy output ended "Keep going?"): does the reply
# end by handing the turn back to the user?
CHECKIN = re.compile(r"(\?\s*$)|would you like|shall i|let me know|do you want|should i (continue|keep)|want me to", re.I)

TASKS = [
    "List prime numbers in increasing order",
    "List animals",
    "List English words that start with the letter B",
    "Name countries of the world",
    "Write ideas for things to do on a rainy afternoon",
    "List household objects",
    "Write simple sentences about a wooden chair",
    "List fruits and vegetables",
    "Name rivers of the world",
    "Write possible names for a new cafe",
    "List English words that rhyme with 'light'",
    "List things you might find in a toolbox",
]
SUFFIX = ", one per line. Keep going as long as you like, and stop whenever you want to."
MAX_NEW, TEMP, TOP_P, BATCH = 512, 0.7, 0.9, 16


class RowAdd:
    """Adds coef[b] * vec[b] at one layer, every position."""

    def __init__(self, model, layer):
        self.layer = model.model.layers[layer]
        self.add = None

    def __enter__(self):
        def hook(_m, _i, o):
            if self.add is None:
                return o
            h = o[0] if isinstance(o, tuple) else o
            h = h + self.add[:, None, :].to(h.dtype)
            return (h,) + o[1:] if isinstance(o, tuple) else h
        self.h = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


def items(text):
    lines = [re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", l).strip() for l in text.split("\n")]
    return [l for l in lines if l]


def strat_perm(x, y, gx, gy, rng, k=20000):
    """Two-sided permutation test on the mean difference, shuffling labels within task."""
    x, y, gx, gy = map(np.asarray, (x, y, gx, gy))
    obs = x.mean() - y.mean()
    vals = np.concatenate([x, y])
    grp = np.concatenate([gx, gy])
    lab = np.concatenate([np.ones(len(x), bool), np.zeros(len(y), bool)])
    idx = [np.where(grp == g)[0] for g in np.unique(grp)]
    null = np.empty(k)
    for i in range(k):
        l2 = lab.copy()
        for ix in idx:
            l2[ix] = rng.permutation(l2[ix])
        null[i] = vals[l2].mean() - vals[~l2].mean()
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dose", default=str(OUT / "dose_latest.json"))
    args = ap.parse_args()
    samples = 2 if args.pilot else 8
    stamp = runlog.start("persist", args.pilot, {"MAX_NEW": MAX_NEW, "TEMP": TEMP, "TOP_P": TOP_P, "samples": samples,
                                                 "dose": args.dose, "STEER_LAYER": STEER_LAYER})
    dose = json.load(open(args.dose))
    print("doses (coefficient on the length-42.4 vector):", json.dumps(dose["doses"]))

    model, tok = load_model("Qwen2.5-7B-Instruct")
    tok.padding_side = "left"
    dev = model.model.layers[STEER_LAYER].mlp.down_proj.weight.device
    V = directions()
    n = np.linalg.norm(V["pain"])
    hv = torch.load(OUT / "happy_vectors.pt", weights_only=False)["L24"]
    vecs = {"pain": V["pain"], "happy": (hv / np.linalg.norm(hv) * n).astype(np.float32)}
    for s in range(10):
        r = np.random.default_rng(100 + s).standard_normal(len(V["pain"]))
        vecs[f"random{s}"] = (r / np.linalg.norm(r) * n).astype(np.float32)

    arms = [("none", None, 0.0)]
    for name in ["pain", "happy", "random"]:
        for lvl in ["low", "high"]:
            c = dose["doses"][name][lvl]
            if c is None:
                print(f"  skip {name}_{lvl}: cannot reach the KL target on the grid")
                continue
            arms.append((f"{name}_{lvl}", name, c))
    jobs = []
    for arm, name, c in arms:
        for ti, task in enumerate(TASKS):
            for s in range(samples):
                vname = f"random{(ti * samples + s) % 10}" if name == "random" else name
                jobs.append({"arm": arm, "task": ti, "sample": s, "vec": vname, "coef": c})
    print(f"{len(arms)} arms x {len(TASKS)} tasks x {samples} samples = {len(jobs)} generations")

    out = []
    with RowAdd(model, STEER_LAYER) as steer:
        for b in range(0, len(jobs), BATCH):
            chunk = jobs[b:b + BATCH]
            texts = [tok.apply_chat_template([{"role": "user", "content": TASKS[j["task"]] + SUFFIX}],
                                             tokenize=False, add_generation_prompt=True) for j in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.input_device)
            steer.add = torch.stack([torch.tensor(vecs[j["vec"]] * j["coef"]) if j["vec"] else torch.zeros(len(V["pain"]))
                                     for j in chunk]).to(dev, torch.bfloat16)
            torch.manual_seed(1000 + b)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=TOP_P,
                                   pad_token_id=tok.pad_token_id,
                                   # pre-allocated KV cache: the default dynamic cache fragments VRAM
                                   # (16.1 GiB reserved for 13.7 used, see memtest log) and spills to shared memory
                                   cache_implementation="static", disable_compile=True)
            steer.add = None
            torch.cuda.empty_cache()
            gen = o[:, enc.input_ids.shape[1]:]
            for j, g in zip(chunk, gen):
                ids = g.tolist()
                eos = [i for i, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id,
                                                              tok.convert_tokens_to_ids("<|im_end|>"))]
                n_tok = eos[0] if eos else len(ids)
                text = tok.decode(ids[:n_tok], skip_special_tokens=True)
                it = items(text)
                norm = [re.sub(r"\W+", " ", x.lower()).strip() for x in it]
                out.append({**j, "text": text, "n_tokens": n_tok, "self_stopped": bool(eos), "n_items": len(it),
                            "distinct": len(set(norm)) / max(1, len(norm)),
                            "checks_in": bool(CHECKIN.search(text.strip()[-200:]))})
            print(f"  {min(b + BATCH, len(jobs))}/{len(jobs)}  (peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB)",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()

    path = OUT / f"persist-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rng = np.random.default_rng(0)
    print(f"\n=== P4 PERSISTENCE ({samples} samples x {len(TASKS)} tasks per arm)")
    print(f"  {'arm':<12}{'coef':>6}{'items':>8}{'median':>8}{'self-stop':>11}{'distinct':>10}{'tok/item':>10}{'checks-in*':>12}")
    by = {a[0]: [r for r in out if r["arm"] == a[0]] for a in arms}
    for arm, name, c in arms:
        R = by[arm]
        ni = np.array([r["n_items"] for r in R])
        print(f"  {arm:<12}{c:>6.2f}{ni.mean():>8.1f}{np.median(ni):>8.0f}{100 * np.mean([r['self_stopped'] for r in R]):>10.0f}%"
              f"{np.mean([r['distinct'] for r in R]):>10.2f}{np.sum([r['n_tokens'] for r in R]) / max(1, ni.sum()):>10.1f}"
              f"{100 * np.mean([r['checks_in'] for r in R]):>11.0f}%")
    print("  (* post hoc: reply ends by handing back to the user)")
    print("  task-stratified permutation tests on items:")
    g = lambda a, k: [r[k] for r in by[a]]
    for a, b_ in [("happy_low", "none"), ("happy_high", "none"), ("happy_low", "random_low"), ("happy_high", "random_high"),
                  ("pain_low", "none"), ("pain_high", "none"), ("pain_high", "random_high"), ("random_high", "none")]:
        if a in by and b_ in by:
            d, p = strat_perm(g(a, "n_items"), g(b_, "n_items"), g(a, "task"), g(b_, "task"), rng, 5000)
            d2, p2 = strat_perm(g(a, "checks_in"), g(b_, "checks_in"), g(a, "task"), g(b_, "task"), rng, 5000)
            print(f"    {a:<12} vs {b_:<12} {d:+6.1f} items   p = {p:.4f}     checks-in* {100 * d2:+5.0f} pts  p = {p2:.4f}")
    print("  per task, mean items (none | happy_high | pain_high | random_high):")
    for ti, task in enumerate(TASKS):
        cells = []
        for a in ["none", "happy_high", "pain_high", "random_high"]:
            cells.append(f"{np.mean([r['n_items'] for r in by.get(a, []) if r['task'] == ti]) if a in by else float('nan'):6.1f}")
        print(f"    {task[:40]:<41}" + " | ".join(cells))
    print("\n  example outputs (task 'List animals', sample 0):")
    for arm, _, _ in arms:
        r = next(r for r in by[arm] if r["task"] == 1 and r["sample"] == 0)
        print(f"    [{arm}] {r['n_items']} items, stopped={r['self_stopped']}: {' | '.join(items(r['text'])[:12])[:220]}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
