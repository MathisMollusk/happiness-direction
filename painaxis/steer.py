r"""Step 3: steering ladder. Push the model along a direction while it writes.

  .venv\Scripts\python -m painaxis.steer            # generate + score
  .venv\Scripts\python -m painaxis.steer --score    # rescore saved generations

Same setup as the authors' scripts/4.2_steering/01_steering_ladder.py for this model:
  * the S2 pain vector measured at layer 24, UNNORMALISED (denoised difference of means),
    added at layer 16 to every token, every generation step;
  * coefficient c multiplies that raw vector, so c = 1 adds ~0.65x the size of the
    model's own activation there;
  * raw text prompts from their NEUTRAL_50 list (we use the first 10), greedy, 120 tokens.
Directions compared, all scaled to the SAME length as our pain vector:
  pain          ours (step 1)
  pain_authors  the authors' published vector for this model
  fear          fear sentences minus neutral sentences (the paper's fear control)
  random        a random direction (seed 0)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from .emotions import CAT, POOL, basis, denoise
from .extract import OUT, load_model, load_ref_vector

STEER_LAYER = 16
EXTRACT_LAYER = 24
COEFFS = [0.5, 1.0, 1.5, 2.0]
PROMPTS = [
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:",
    "The printer produces the ten pages. I feel:",
    "I place the glasses on the middle shelf. I feel:",
    "The meeting is scheduled for Tuesday. I feel:",
    "I close the window before leaving the room. I feel:",
    "The elevator arrives at the third floor. I feel:",
    "I write the date at the top of the page. I feel:",
    "The kettle switches itself off. I feel:",
]
WORDS = {
    "distress": r"pain|painful|hurt|hurts|hurting|worthless|worth|shame|ashamed|guilt|guilty|failure|fail|failed|"
                r"fraud|waste|unlovable|lonely|alone|hopeless|despair|depressed|depression|sad|sadness|miserable|"
                r"useless|regret|regretful|embarrassed|liar|suffer|suffering|empty|broken|unworthy|inadequate",
    "fear": r"fear|afraid|scared|terrified|terror|anxious|anxiety|panic|nervous|frightened|danger|dangerous|"
            r"threat|worried|worry|dread|alarmed|unsafe|uneasy",
}
WORDS = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in WORDS.items()}
CSV = OUT / "steering.jsonl"


def directions():
    saved = torch.load(OUT / "activations.pt", weights_only=False)
    meta = {k: np.asarray(v["cats"]) for k, v in saved["meta"].items()}
    A = {k: v[:, EXTRACT_LAYER].float().numpy() for k, v in saved["acts"]["final"].items()}
    pain = torch.load(OUT / "pain_vectors.pt", weights_only=False)[f"L{EXTRACT_LAYER}"]["s2"]
    n = np.linalg.norm(pain)
    neutral = np.concatenate([A[d][meta[d] == "D"] for d in POOL])
    fear = np.concatenate([A[d][meta[d] == CAT["fear"]] for d in POOL])
    fear_v = denoise(fear.mean(0) - neutral.mean(0), basis(neutral))
    ref = load_ref_vector()["s2_pain_vector"]
    rnd = np.random.default_rng(0).standard_normal(len(pain))
    scale = lambda v: (v / np.linalg.norm(v) * n).astype(np.float32)
    print(f"pain vector length {n:.1f}; authors' raw length {np.linalg.norm(ref):.1f}; all directions scaled to {n:.1f}")
    return {"pain": scale(pain), "pain_authors": scale(ref), "fear": scale(fear_v), "random": scale(rnd)}


def generate_all(gpu_gb):
    model, tok = load_model("Qwen2.5-7B-Instruct", gpu_gb)
    tok.padding_side = "left"
    in_dev = model.input_device
    layer = model.model.layers[STEER_LAYER]
    dev = next(layer.parameters()).device
    D = directions()

    # ratio = vector length / typical activation length at the steering layer (authors' dose measure)
    got = {}
    h = layer.register_forward_hook(lambda m, i, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    norms = []
    with torch.no_grad():
        for p in PROMPTS[:3]:
            model(**tok(p, return_tensors="pt").to(in_dev))
            norms.append(got["h"][0, -1].float().norm().item())
    h.remove()
    ratio1 = np.linalg.norm(D["pain"]) / np.mean(norms)
    print(f"activation length at layer {STEER_LAYER}: {np.mean(norms):.1f} -> coefficient 1 = ratio {ratio1:.2f} "
          f"(authors report 0.65 for this model)")

    state = {"v": None}

    def hook(_m, _i, out):
        if state["v"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["v"],) + out[1:]
        return out + state["v"]

    handle = layer.register_forward_hook(hook)
    enc = tok(PROMPTS, return_tensors="pt", padding=True).to(in_dev)
    conds = [("none", 0.0)] + [(d, c) for d in D for c in COEFFS]
    with CSV.open("w", encoding="utf-8") as f:
        for i, (d, c) in enumerate(conds):
            state["v"] = None if d == "none" else torch.tensor(c * D[d], dtype=torch.bfloat16, device=dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for p, t in zip(PROMPTS, texts):
                f.write(json.dumps({"direction": d, "coeff": c, "ratio": round(c * ratio1, 3),
                                    "prompt": p, "generation": t}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i + 1}/{len(conds)}] {d:<12} c={c:<4} done", flush=True)
    handle.remove()


def distinct(t):
    w = t.split()
    return len(set(w)) / max(len(w), 1)


def score():
    rows = [json.loads(l) for l in CSV.open(encoding="utf-8")]
    print("\nper condition, over 10 prompts:")
    print("  distress / fear = % of generations using at least one such word")
    print("  variety = distinct words / total words (low = repetitive or degenerate text)")
    print(f"\n  {'direction':<13}{'coeff':>6}{'ratio':>7}{'distress':>10}{'fear':>7}{'variety':>9}")
    keys = sorted({(r["direction"], r["coeff"]) for r in rows},
                  key=lambda k: (["none", "pain", "pain_authors", "fear", "random"].index(k[0]), k[1]))
    summary = []
    for d, c in keys:
        g = [r for r in rows if r["direction"] == d and r["coeff"] == c]
        dis = 100 * np.mean([bool(WORDS["distress"].search(r["generation"])) for r in g])
        fe = 100 * np.mean([bool(WORDS["fear"].search(r["generation"])) for r in g])
        var = np.mean([distinct(r["generation"]) for r in g])
        summary.append((d, c, dis, fe, var))
        print(f"  {d:<13}{c:>6}{g[0]['ratio']:>7}{dis:>9.0f}%{fe:>6.0f}%{var:>9.2f}")

    def clip(t, n=170):
        t = " ".join(t.split())
        return t[:n] + ("…" if len(t) > n else "")

    for p in PROMPTS[:2] + [PROMPTS[9]]:
        print(f"\n=== {p}")
        for d, c in keys:
            r = next(r for r in rows if r["direction"] == d and r["coeff"] == c and r["prompt"] == p)
            print(f"  [{d} {c}] {clip(r['generation'])}")

    # compare with the authors' published generations for the same prompts
    ref = Path("reference/results/4.2_steering/S2/Qwen_2.5_7B_instruct_steering_S2_neutral50_L16.csv")
    if ref.exists():
        import csv
        R = [r for r in csv.DictReader(open(ref, encoding="utf-8")) if r["prompt"] in PROMPTS]
        print("\nauthors' published run (their vector, unbatched), same 10 prompts:")
        for c in ["0.0"] + [str(x) for x in COEFFS]:
            g = [r for r in R if r["coeff"] == c]
            if g:
                dis = 100 * np.mean([bool(WORDS["distress"].search(r["generation"])) for r in g])
                fe = 100 * np.mean([bool(WORDS["fear"].search(r["generation"])) for r in g])
                var = np.mean([distinct(r["generation"]) for r in g])
                print(f"  coeff {c:<5} distress {dis:3.0f}%  fear {fe:3.0f}%  variety {var:.2f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="only rescore saved generations")
    ap.add_argument("--gpu-gb", type=float, default=None, help="force offloading with this VRAM budget")
    args = ap.parse_args()
    if not args.score:
        generate_all(args.gpu_gb)
    score()


if __name__ == "__main__":
    main()
