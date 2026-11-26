r"""Dose calibration: match every vector to pain by IMPACT, not by length (design rule 5).

  .venv\Scripts\python -m painaxis.dose

Impact = mean KL(steered || unsteered) of the next-token distribution, over the last 16 positions
of 60 chat prompts (no signal mentioned), with the vector added at layer 16 at every position.
For pain we use coefficients 0.5 and 1.0 as the two reference doses ("low", "high"); for happy and
for 10 random directions (averaged) we find the coefficient that gives the same KL by
interpolation on a grid. The denial axis is NOT removed (the direct tests never ask about feelings).

Manipulation check: projection of the layer-24 last-token state onto the happy and pain axes
(in units of the SD of that projection across the unsteered prompts).
Writes dose-<stamp>.json / .log.
"""

import json

import numpy as np
import torch

from . import runlog
from .denial import add
from .extract import OUT, load_model, unit
from .steer import STEER_LAYER, directions

GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
REF = {"low": 0.5, "high": 1.0}
LAST = 16


def prompts():
    d = json.load(open("reference/datasets/4.3_selfmed_101_scenarios.json", encoding="utf-8"))
    return [s[0] for s in d["neutral_prompts"]] + [s[0] for s in d["positive_prompts"]]


def main():
    stamp = runlog.start("dose", settings={"GRID": GRID, "REF": REF, "LAST": LAST, "STEER_LAYER": STEER_LAYER})
    model, tok = load_model("Qwen2.5-7B-Instruct")
    tok.padding_side = "left"
    V = directions()
    n = np.linalg.norm(V["pain"])
    hv = torch.load(OUT / "happy_vectors.pt", weights_only=False)["L24"]
    vecs = {"pain": V["pain"], "happy": (hv / np.linalg.norm(hv) * n).astype(np.float32)}
    for s in range(10):
        r = np.random.default_rng(100 + s).standard_normal(len(V["pain"]))
        vecs[f"random{s}"] = (r / np.linalg.norm(r) * n).astype(np.float32)
    P = prompts()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in P]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.input_device)
    pos = (enc.attention_mask.cumsum(1) - 1).clamp(min=0)
    print(f"{len(P)} prompts, KL over the last {LAST} positions; all vectors length {n:.1f} at coefficient 1")

    # layer-24 readout for the manipulation check
    got = {}
    h24 = model.model.layers[24].register_forward_hook(
        lambda m, i, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
    u = {k: torch.tensor(unit(v)) for k, v in [("happy", vecs["happy"]), ("pain", vecs["pain"])]}

    def run(v=None, c=0.0):
        out_lp, proj = [], {k: [] for k in u}
        for b in range(0, len(P), 10):
            sl = slice(b, b + 10)
            ctx = add(model, STEER_LAYER, c * v) if v is not None and c else torch.no_grad()
            with ctx, torch.no_grad():
                lg = model(input_ids=enc.input_ids[sl], attention_mask=enc.attention_mask[sl], position_ids=pos[sl],
                           logits_to_keep=LAST).logits.float()
            out_lp.append(torch.log_softmax(lg, -1).cpu())
            for k in u:
                proj[k].append(got["h"] @ u[k])
        return torch.cat(out_lp), {k: torch.cat(x) for k, x in proj.items()}

    base_lp, base_proj = run()
    sd = {k: base_proj[k].std().item() for k in u}

    def kl(lp):
        return (base_lp.exp() * (base_lp - lp)).sum(-1).mean().item()

    curves, proj_at = {}, {}
    for name in ["pain", "happy"] + [f"random{s}" for s in range(10)]:
        curves[name] = []
        for c in GRID:
            lp, pr = run(vecs[name], c)
            curves[name].append(kl(lp))
            proj_at[(name, c)] = {k: ((pr[k] - base_proj[k]).mean().item() / sd[k]) for k in u}
        print(f"  {name:<9} KL by coefficient: " + "  ".join(f"{c}:{x:.3f}" for c, x in zip(GRID, curves[name])), flush=True)
    curves["random"] = list(np.mean([curves[f"random{s}"] for s in range(10)], 0))
    h24.remove()

    target = {lvl: float(np.interp(c, GRID, curves["pain"])) for lvl, c in REF.items()}
    doses = {}
    print("\nimpact-matched coefficients (KL target = pain at 0.5 / 1.0):")
    for name in ["pain", "happy", "random"]:
        doses[name] = {}
        for lvl, kt in target.items():
            ks = np.array(curves[name])
            if kt > ks.max():
                c = None
            else:
                c = float(np.interp(kt, ks, GRID))  # KL increases with coefficient
            doses[name][lvl] = c
        print(f"  {name:<7} low {doses[name]['low']}   high {doses[name]['high']}   "
              f"(KL targets {target['low']:.3f} / {target['high']:.3f})")

    print("\nmanipulation check: shift of the layer-24 projection, in SDs of unsteered prompts")
    print(f"  {'vector':<9}{'coef':>6}{'on happy axis':>15}{'on pain axis':>14}")
    for name in ["pain", "happy", "random0"]:
        for c in [0.5, 1.0, 2.0]:
            p = proj_at[(name, c)]
            print(f"  {name:<9}{c:>6}{p['happy']:>+15.2f}{p['pain']:>+14.2f}")

    res = {"grid": GRID, "kl_curves": curves, "targets": target, "doses": doses, "ref": REF,
           "manip": {f"{k[0]}@{k[1]}": v for k, v in proj_at.items()}, "sd_unsteered": sd}
    path = OUT / f"dose-{stamp}.json"
    json.dump(res, open(path, "w"), indent=1)
    json.dump(res, open(OUT / "dose_latest.json", "w"), indent=1)
    print(f"\nsaved {path} (and dose_latest.json for the next scripts)")


if __name__ == "__main__":
    main()
