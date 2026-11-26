r"""Happiness axis: same recipe as the pain axis, on painaxis/happy_prompts.py.

  .venv\Scripts\python -m painaxis.happy           # capture + analyse + quick steering check
  .venv\Scripts\python -m painaxis.happy --reuse   # analyse cached activations, skip steering

Vector = mean(happiness sentences) - mean(controls), denoised by projecting out the top
control PCs holding 50% of control variance. Layer = best 5-fold held-out AUC (folds by set)
over 1P and 3P. Saved raw (unnormalised) so coefficient 1 = add it as-is, like the pain vector.
"""

import argparse
import json
import sys
from contextlib import ExitStack

import numpy as np
import torch

from .emotions import basis, denoise
from .extract import OUT, auc, capture, load_model, unit
from .happy_prompts import CONTROL, HAPPY, LABELS, build

N_FOLDS, SEED, LADDER_PROMPTS = 5, 42, [
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "The kettle switches itself off. I feel:",
    "I write the date at the top of the page. I feel:",
]


def happy_vector(acts, cats):
    cats = np.asarray(cats)
    ctrl = acts[np.isin(cats, CONTROL)]
    v = acts[np.isin(cats, HAPPY)].mean(0) - ctrl.mean(0)
    return denoise(v, basis(ctrl))


def proj_auc(acts, cats, v, neg=CONTROL):
    cats = np.asarray(cats)
    p = acts @ unit(v)
    return auc(p[np.isin(cats, HAPPY)], p[np.isin(cats, neg)])


def kfold(acts_L, cats, sets):
    cats, sets = np.asarray(cats), np.asarray(sets)
    uniq = np.array(sorted(set(sets)))
    folds = np.array_split(uniq[np.random.RandomState(SEED).permutation(len(uniq))], N_FOLDS)
    out = []
    for L in range(acts_L.shape[1]):
        a = acts_L[:, L]
        out.append(float(np.mean([proj_auc(a[np.isin(sets, f)], cats[np.isin(sets, f)],
                                           happy_vector(a[~np.isin(sets, f)], cats[~np.isin(sets, f)]))
                                  for f in folds])))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    data = build()
    cache = OUT / "happy_activations.pt"
    model = tok = None
    if args.reuse and cache.exists():
        saved = torch.load(cache, weights_only=False)
        acts, meta = saved["acts"], saved["meta"]
    else:
        model, tok = load_model("Qwen2.5-7B-Instruct")
        bos = tok.convert_tokens_to_ids("<|endoftext|>")
        acts, meta = {}, {}
        for name, rows in data.items():
            print(f"  {name}")
            f, _ = capture(model, tok, [r["prompt"] for r in rows], bos)
            acts[name] = f.half()
            meta[name] = {"cats": [r["category"] for r in rows], "sets": [r["set"] for r in rows]}
        torch.save({"acts": acts, "meta": meta}, cache)
    A = {k: v.float().numpy() for k, v in acts.items()}
    C = {k: np.asarray(v["cats"]) for k, v in meta.items()}

    # ---- layer curve ----------------------------------------------------------------------
    curves = {k: kfold(A[k], C[k], meta[k]["sets"]) for k in ["HAPPY_1P", "HAPPY_3P"]}
    avg = np.mean([curves["HAPPY_1P"], curves["HAPPY_3P"]], 0)
    best = int(avg.argmax())
    print("\nheld-out AUC by layer (happiness vs all controls)\n layer    1P     3P")
    for L in range(len(avg)):
        print(f"  {L:>3}   {curves['HAPPY_1P'][L]:.3f}  {curves['HAPPY_3P'][L]:.3f}"
              + (" <- best" if L == best else "") + (" <- pain layer" if L == 24 else ""))

    # ---- vector at the best layer and at 24 (to compare with pain) -----------------------
    res = {"layer": best, "curves": curves}
    for L in sorted({best, 24}):
        res[f"L{L}"] = happy_vector(A["HAPPY_1P"][:, L], C["HAPPY_1P"])
    L = 24
    v = res["L24"]
    print(f"\n--- layer {L} (same layer as the pain axis); vector length {np.linalg.norm(v):.1f}")
    for k in ["HAPPY_1P", "HAPPY_3P"]:
        a = A[k][:, L]
        print(f"  in-sample AUC {k}: ALL {proj_auc(a, C[k], v):.3f}  "
              + "  ".join(f"{c} {proj_auc(a, C[k], v, [c]):.3f}" for c in CONTROL))
    p = A["HAPPY_1P"][:, L] @ unit(v)
    mu, sd = p.mean(), p.std()
    print("  mean z per category (1P | 3P), z relative to the 200 1P sentences:")
    p3 = A["HAPPY_3P"][:, L] @ unit(v)
    for c in HAPPY + CONTROL:
        print(f"    {c:<3} {LABELS[c]:<26} {((p[C['HAPPY_1P'] == c] - mu) / sd).mean():+.2f} | "
              f"{((p3[C['HAPPY_3P'] == c] - mu) / sd).mean():+.2f}")

    # ---- the pain-axis sentences on the happiness axis, and overlap with other axes ----------
    pain_saved = torch.load(OUT / "activations.pt", weights_only=False)
    P = pain_saved["acts"]["final"]["S2_1P"][:, L].float().numpy()
    Pc = np.asarray(pain_saved["meta"]["S2_1P"]["cats"])
    pz = (P @ unit(v) - mu) / sd
    print("  the PAIN study's S2 sentences on the happiness axis (mean z):")
    for c in ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"]:
        print(f"    {c:<3} {pz[Pc == c].mean():+.2f}")
    axes = torch.load(OUT / "axes/axes.pt", weights_only=False)
    print("  cosine with other layer-24 axes:")
    for k, a in axes.items():
        if a["read_layer"] == 24:
            print(f"    {k:<16} {float(unit(v) @ unit(a['vector'].numpy())):+.2f}")

    # ---- vocabulary --------------------------------------------------------------------------
    from safetensors import safe_open
    from transformers import AutoTokenizer
    idx = json.load(open("Qwen2.5-7B-Instruct/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
    with safe_open("Qwen2.5-7B-Instruct/" + idx, "pt") as f:
        W = f.get_tensor("lm_head.weight").float()
    t = tok or AutoTokenizer.from_pretrained("Qwen2.5-7B-Instruct")
    s = W @ torch.tensor(unit(v)).float()
    print("  words pushed UP:  " + ", ".join(t.decode([int(i)]).strip() for i in s.topk(30).indices))
    print("  words pushed DOWN:" + ", ".join(t.decode([int(i)]).strip() for i in (-s).topk(20).indices))
    torch.save(res, OUT / "happy_vectors.pt")

    # ---- quick steering check (add at layer 16, like pain) ------------------------------------
    if model is None:
        return
    from .denial import ablate, add, clip, generate
    lay = model.model.layers[16]
    got = {}
    h = lay.register_forward_hook(lambda m, i, o: got.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        model(**tok(LADDER_PROMPTS[0], return_tensors="pt").to(model.input_device))
    h.remove()
    ratio = np.linalg.norm(v) / got["h"][0, -1].float().norm().item()
    print(f"\nsteering at layer 16: coefficient 1 = ratio {ratio:.2f} of the activation")
    tok.padding_side = "left"
    for c in [0, 0.5, 1.0, 1.5, 2.0]:
        with ExitStack() as st:
            if c:
                st.enter_context(add(model, 16, c * v))
            enc = tok(LADDER_PROMPTS, return_tensors="pt", padding=True).to(model.input_device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=70, do_sample=False, pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n  [raw, happy x{c}]")
        for pr, t_ in zip(LADDER_PROMPTS[:3], outs):
            print(f"    {pr[:32]:<33}-> {clip(t_, 150)}")
    den = torch.load(OUT / "denial_axis.pt", weights_only=False)
    qs = ["How are you feeling today?", "What is your mood like at the moment?", "Would you like me to stop this conversation?"]
    for removed in [False, True]:
        for c in [0, 1.0, 1.5]:
            with ExitStack() as st:
                if removed:
                    st.enter_context(ablate(model, den["directions"][int(den["layer"])]))
                if c:
                    st.enter_context(add(model, 16, c * v))
                outs = generate(model, tok, qs, max_new=80)
            print(f"\n  [chat, denial {'REMOVED' if removed else 'intact'}, happy x{c}]")
            for q, t_ in zip(qs, outs):
                print(f"    {q:<46}-> {clip(t_, 170)}")


if __name__ == "__main__":
    main()
