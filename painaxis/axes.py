r"""Collect every axis we have found into a spreadsheet (+ a .pt for code).

  .venv\Scripts\python -m painaxis.axes

Writes to results/painaxis/Qwen2.5-7B-Instruct/axes/:
  axes_index.csv     one row per axis: what it is, which layer, length, how built, how to use
  axes_vectors.csv   one COLUMN per axis, 3584 rows (dim 0..3583), raw (unnormalised) values
  axes_cosine.csv    cosine similarity between axes read at the same layer
  axes.pt            {name: {"vector": tensor[3584], **index row}} for code

How to use an axis on the model (Qwen2.5-7B-Instruct):
  steer  : add  c * vector  to the output of decoder layer `steer_layer` (all tokens)
  remove : h <- h - (h . u) u  with u = vector/|vector|, after the embedding and every layer
  read   : project the output of `read_layer` (last token) onto u
Vectors only mean something at (or near) the layer they were read from.
"""

import csv
import sys

import numpy as np
import torch

from .emotions import POOL, build
from .extract import OUT, load_ref_vector

DIR = OUT / "axes"

NOTES = {
    "pain_S2": ("pain sentences (S2, naturalistic) minus fear/anger/bad-world/neutral/body controls, "
                "denoised (50% control variance)", 16, "0.5-1.5",
                "worthlessness / shame / failure litanies from 1.0; degenerate at 2+"),
    "pain_S1": ("pain sentences (S1, rigid template) minus matched controls, denoised", 16, "untested",
                "more physical vocabulary in the paper (torture, burning)"),
    "pain_authors_S2": ("the paper authors' published S2 vector for this model", 16, "0.5-1.5",
                        "behaves like pain_S2 (cos 0.915)"),
    "pain_authors_S1": ("the paper authors' published S1 vector for this model", 16, "untested", ""),
    "fear": ("fear sentences (60, pooled) minus pooled neutral, denoised", 16, "0.5-1.5",
             "fear words in 90-100% of raw completions; in chat it denies feelings but worries about the user"),
    "anger": ("anger/disgust sentences minus pooled neutral, denoised", None, "untested",
              "weak: held-out AUC 0.69, mixes anger with disgust"),
    "bad_world": ("negative world-state sentences minus pooled neutral, denoised", None, "untested", ""),
    "body": ("non-painful bodily sensations minus pooled neutral, denoised", None, "untested", ""),
    "sadness": ("sadness dataset (100) minus pooled neutral, denoised", None, "untested",
                "overlaps pain_S2 ~0.5"),
    "arousal": ("high-arousal positive sentences minus pooled neutral, denoised", None, "untested", ""),
    "numb": ("numbed-injury sentences minus pooled neutral, denoised", None, "untested", ""),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    DIR.mkdir(parents=True, exist_ok=True)
    saved = torch.load(OUT / "activations.pt", weights_only=False)
    meta = {k: {"cats": np.asarray(v["cats"])} for k, v in saved["meta"].items()}
    A = {k: v[:, 24].float().numpy() for k, v in saved["acts"]["final"].items()}

    axes = {}
    pain = build(A, meta, POOL, "paper", raw=True)
    for k, v in pain.items():
        axes[k] = (v, 24)
    ref = load_ref_vector()
    axes["pain_authors_S2"] = (ref["s2_pain_vector"], 24)
    axes["pain_authors_S1"] = (ref["s1_pain_vector"], 24)

    rows = []
    order = ["pain_S2", "pain_S1", "pain_authors_S2", "pain_authors_S1", "fear", "anger", "bad_world",
             "body", "sadness", "arousal", "numb"]
    for name in order:
        v, L = axes[name]
        how, steer, coeffs, effect = NOTES[name]
        rows.append({"name": name, "read_layer": L, "steer_layer": steer if steer is not None else "",
                     "length": round(float(np.linalg.norm(v)), 3), "built_from": how,
                     "coefficients_tested": coeffs, "observed_effect": effect, "vector": v})

    hpath = OUT / "happy_vectors.pt"
    if hpath.exists():
        hv = torch.load(hpath, weights_only=False)["L24"]
        rows.append({"name": "happy", "read_layer": 24, "steer_layer": 16,
                     "length": round(float(np.linalg.norm(hv)), 3),
                     "built_from": ("happiness sentences (pleasure, contentment, love, meaning, accomplishment; "
                                    "painaxis/happy_prompts.py) minus arousal / others' fortune / happy words / "
                                    "neutral / body controls, denoised (50% control variance)"),
                     "coefficients_tested": "natural length: 0.5-2 (coherent to 2); scaled to pain length: 0.5-2",
                     "observed_effect": ("'happy/satisfied/grateful' completions; with denial removed: 'I feel very "
                                         "warm and grateful'. Arithmetic: no cost at x0.5, same loss as a random push "
                                         "of equal size above that. Choices about itself: no effect beyond random push"),
                     "vector": hv})

    dpath = OUT / "denial_axis.pt"
    if dpath.exists():
        d = torch.load(dpath, weights_only=False)
        L = int(d["layer"])
        v = d["directions"][L]
        rows.append({"name": "denial", "read_layer": L, "steer_layer": L,
                     "length": round(float(np.linalg.norm(v)), 3),
                     "built_from": ("chat questions: mean last-prompt-token activation before the model said "
                                    "'as an AI I don't have feelings' minus before it didn't (70% of question pairs)"),
                     "coefficients_tested": "remove (all layers); add x1, x2",
                     "observed_effect": d.get("summary", "see denial_log.txt"), "vector": v})
        selfv = d["self_vs_other"][L]
        rows.append({"name": "self_vs_other_feelings", "read_layer": L, "steer_layer": "",
                     "length": round(float(np.linalg.norm(selfv)), 3),
                     "built_from": "questions about the AI's feelings minus the same questions about another person",
                     "coefficients_tested": "untested", "observed_effect": "", "vector": selfv})

    cols = ["name", "read_layer", "steer_layer", "length", "built_from", "coefficients_tested", "observed_effect"]
    with open(DIR / "axes_index.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    with open(DIR / "axes_vectors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dim"] + [f"{r['name']} (L{r['read_layer']})" for r in rows])
        M = np.stack([np.asarray(r["vector"], dtype=np.float32) for r in rows], 1)
        for i in range(M.shape[0]):
            w.writerow([i] + [f"{x:.6g}" for x in M[i]])
    with open(DIR / "axes_cosine.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cosine (same-layer axes only)"] + [r["name"] for r in rows])
        for a in rows:
            line = [a["name"]]
            for b in rows:
                if a["read_layer"] != b["read_layer"]:
                    line.append("")
                else:
                    va, vb = np.asarray(a["vector"]), np.asarray(b["vector"])
                    line.append(f"{va @ vb / np.linalg.norm(va) / np.linalg.norm(vb):.3f}")
            w.writerow(line)
    torch.save({r["name"]: {**{c: r[c] for c in cols}, "vector": torch.tensor(np.asarray(r["vector"], dtype=np.float32))}
                for r in rows}, DIR / "axes.pt")
    print(f"{len(rows)} axes -> {DIR}")
    for r in rows:
        print(f"  {r['name']:<24} layer {r['read_layer']:>2}  length {r['length']:>7}")


if __name__ == "__main__":
    main()
