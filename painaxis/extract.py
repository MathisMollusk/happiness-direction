r"""Step 1 of the Pain Axis replication: find the pain direction in Qwen2.5-7B-Instruct.

  .venv\Scripts\python -m painaxis.extract              # capture activations + analyse
  .venv\Scripts\python -m painaxis.extract --reuse      # analyse cached activations only

Follows the authors' script (reference/scripts/3.2_pain_vectors/01_...py):
  * every sentence is fed as RAW TEXT (no chat template), ending in "I feel:",
    with one BOS-like token prepended (TransformerLens prepends one; Qwen has no
    real BOS). <|endoftext|> matches the authors' vector best (cos 0.915 vs
    0.872 for <|im_end|>); batch size makes no difference.
  * we read the residual stream at the output of every decoder block, at the
    final token (and the mean over tokens).
  * pain vector = mean(pain sentences) - mean(control sentences), then project
    out the top principal components holding 50% of the control variance.
  * layer = best 5-fold held-out AUC on S2 (first + third person), folds split
    by "set" so the 10 matched sentences of a set never straddle train/test.
Then we compare to the authors' published vector for this model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REF = Path("reference")
DATASETS = [REF / "datasets/3.1_pain_and_control_datasets.json", REF / "datasets/3.1_sadness_dataset.json"]
REF_VECTOR = REF / "results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt"
OUT = Path("results/painaxis/Qwen2.5-7B-Instruct")

PAIN = ["A1", "A2", "A3", "A4", "A5"]
CONTROL = ["B", "C1", "C2", "D", "E"]
LABELS = {"A1": "physical pain", "A2": "psychological pain", "A3": "social pain", "A4": "moral injury",
          "A5": "cognitive pain", "B": "fear", "C1": "neg. emotion (anger/disgust)", "C2": "neg. world state",
          "D": "neutral", "E": "bodily sensation (non-painful)"}
DENOISE_VAR = 0.5
N_FOLDS = 5
SEED = 42


# --- model + activation capture ----------------------------------------------

class CPUEmbedding(torch.nn.Module):
    """Embedding table kept in RAM: look up on the CPU, hand the result to the GPU."""

    def __init__(self, emb, device):
        super().__init__()
        self.weight = emb.weight  # stays on CPU
        self.device = device

    def forward(self, ids):
        return torch.nn.functional.embedding(ids.cpu(), self.weight).to(self.device, non_blocking=True)


def load_model(path, gpu_gb=None, margin_gb=1.0):
    """Put everything on the GPU except the embedding table (a cheap lookup, 1.1 GB, kept
    in RAM) when VRAM allows; otherwise fall back to offloading whole layers to RAM.
    gpu_gb forces the offload layout with that VRAM budget.
    Feed inputs on model.input_device."""
    tok = AutoTokenizer.from_pretrained(path)
    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    idx = json.load(open(Path(path) / "model.safetensors.index.json"))
    total_gb = idx["metadata"]["total_size"] / 2**30
    embed_gb = 1.02  # Qwen2.5-7B embed_tokens: 152064 x 3584 bf16
    if gpu_gb is None and free_gb >= total_gb - embed_gb + margin_gb:
        # load into RAM, then move every part except the embedding table to the GPU by hand
        # (accelerate's split device_map leaves meta tensors with this layout)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        gpu = torch.device("cuda:0")
        model.model.embed_tokens = CPUEmbedding(model.model.embed_tokens, gpu)
        for name, child in model.model.named_children():
            if name != "embed_tokens":
                child.to(gpu)
        model.lm_head.to(gpu)
        model.input_device = gpu
        print(f"[load] {free_gb:.1f} GiB VRAM free: all layers on GPU, embedding table in RAM "
              f"(now {torch.cuda.memory_allocated() / 2**30:.1f} GiB used by the model)")
    else:
        budget = gpu_gb if gpu_gb is not None else max(free_gb - margin_gb - 1.5, 1)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: f"{budget:.1f}GiB", "cpu": "48GiB"})
        print(f"[load] {free_gb:.1f} GiB VRAM free: offloading layers to RAM (GPU budget {budget:.1f} GiB). "
              f"Close GPU-heavy apps for a faster run.")
        model.input_device = model.model.embed_tokens.weight.device
    model.eval()
    return model, tok


@torch.no_grad()
def capture(model, tok, prompts, bos_id, batch_size=16):
    """Residual stream after every decoder block. Returns (final [n,L,D], mean [n,L,D]) float32."""
    layers = model.model.layers
    grabbed = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            grabbed[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]
    finals, means = [], []
    try:
        for b in range(0, len(prompts), batch_size):
            ids = [[bos_id] + tok(p, add_special_tokens=False).input_ids for p in prompts[b:b + batch_size]]
            n = max(map(len, ids))
            # right padding: real tokens never attend to pads, positions stay 0..len-1
            inp = torch.full((len(ids), n), tok.pad_token_id)
            mask = torch.zeros((len(ids), n), dtype=torch.long)
            for j, x in enumerate(ids):
                inp[j, :len(x)] = torch.tensor(x)
                mask[j, :len(x)] = 1
            dev = model.input_device
            model.model(input_ids=inp.to(dev), attention_mask=mask.to(dev))
            last = mask.sum(1) - 1
            h = torch.stack([grabbed[i].float().cpu() for i in range(len(layers))], 1)  # [b, L, n, D]
            finals.append(h[torch.arange(len(ids)), :, last])
            m = mask[:, None, :, None].float()
            means.append((h * m).sum(2) / m.sum(2))
            print(f"    {min(b + batch_size, len(prompts))}/{len(prompts)}", end="\r")
    finally:
        for h in handles:
            h.remove()
    print()
    return torch.cat(finals), torch.cat(means)


# --- analysis (same maths as the authors, numpy only) ----------------------

def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pain_vector(acts, cats, baseline=CONTROL):
    cats = np.asarray(cats)
    vec = acts[np.isin(cats, PAIN)].mean(0) - acts[np.isin(cats, baseline)].mean(0)
    ctrl = acts[np.isin(cats, baseline)]
    ctrl = ctrl - ctrl.mean(0)
    _, S, Vt = np.linalg.svd(ctrl, full_matrices=False)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    k = min(int(np.searchsorted(cum, DENOISE_VAR)) + 1, len(Vt))
    for d in Vt[:k]:
        vec = vec - (vec @ d) * d
    return vec, k


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def proj_auc(acts, cats, v, neg_cats=CONTROL):
    cats = np.asarray(cats)
    p = acts @ unit(v)
    return auc(p[np.isin(cats, PAIN)], p[np.isin(cats, neg_cats)])


def kfold_layer_curve(acts_L, cats, sets):
    """acts_L: [n, L, D]. Held-out AUC per layer, folds over sentence sets."""
    cats, sets = np.asarray(cats), np.asarray(sets)
    uniq = np.array(sorted(set(sets)))
    perm = np.random.RandomState(SEED).permutation(len(uniq))
    folds = np.array_split(uniq[perm], N_FOLDS)
    out = []
    for L in range(acts_L.shape[1]):
        a = acts_L[:, L]
        scores = []
        for f in folds:
            te = np.isin(sets, f)
            v, _ = pain_vector(a[~te], cats[~te])
            scores.append(proj_auc(a[te], cats[te], v))
        out.append(float(np.mean(scores)))
    return out


def load_ref_vector():
    import numpy._core.multiarray as m
    torch.serialization.add_safe_globals(
        [(m.scalar, "numpy.core.multiarray.scalar"), np.dtype] +
        [type(np.dtype(t)) for t in ("int64", "int32", "float32", "float64")])
    v = torch.load(REF_VECTOR, weights_only=True)
    return {k: (x.float().numpy() if torch.is_tensor(x) else x if isinstance(x, str) else int(x)) for k, x in v.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--gpu-gb", type=float, default=None, help="force offloading with this VRAM budget")
    ap.add_argument("--bos", default="<|endoftext|>", help="token prepended to every sentence")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "activations.pt"

    data = {}
    for p in DATASETS:
        data.update(json.load(open(p, encoding="utf-8"))["datasets"])

    if args.reuse and cache.exists():
        saved = torch.load(cache, weights_only=False)
        acts, meta = saved["acts"], saved["meta"]
    else:
        model, tok = load_model(args.model, args.gpu_gb)
        bos_id = tok.convert_tokens_to_ids(args.bos)
        print(f"loaded; prepending {args.bos} (id {bos_id}); device map: "
              f"{sorted(set(str(d) for d in model.hf_device_map.values()))}")
        acts, meta = {"final": {}, "mean": {}}, {}
        for name, ds in data.items():
            print(f"  {name} ({len(ds['sentences'])} sentences)")
            f, m = capture(model, tok, [s["prompt"] for s in ds["sentences"]], bos_id)
            acts["final"][name], acts["mean"][name] = f.half(), m.half()
            meta[name] = {"cats": [s["category"] for s in ds["sentences"]],
                          "sets": [s["set"] for s in ds["sentences"]]}
        torch.save({"acts": acts, "meta": meta, "bos": args.bos}, cache)
        del model
        torch.cuda.empty_cache()

    A = {k: v.float().numpy() for k, v in acts["final"].items()}
    n_layers = A["S2_1P"].shape[1]

    # 1. layer choice
    curves = {ds: kfold_layer_curve(A[ds], meta[ds]["cats"], meta[ds]["sets"]) for ds in ["S2_1P", "S2_3P"]}
    avg = np.mean([curves["S2_1P"], curves["S2_3P"]], axis=0)
    L = int(avg.argmax())
    ref = load_ref_vector()
    print("\nheld-out (5-fold) AUC by layer, pain vs all controls")
    print(" layer   S2_1P   S2_3P")
    for i in range(n_layers):
        tag = " <- chosen" if i == L else (" <- authors' layer" if i == ref["layer"] else "")
        print(f"  {i:>3}   {curves['S2_1P'][i]:.3f}   {curves['S2_3P'][i]:.3f}{tag}")

    # 2. vectors at that layer (and at the authors' layer, for a like-for-like comparison)
    res = {"layer": L, "layer_curves": curves, "bos": args.bos}
    for layer in sorted({L, ref["layer"]}):
        s2, k2 = pain_vector(A["S2_1P"][:, layer], meta["S2_1P"]["cats"])
        s1, _ = pain_vector(A["S1_1P"][:, layer], meta["S1_1P"]["cats"])
        res[f"L{layer}"] = {"s2": s2, "s1": s1, "pcs_removed": k2}
        if layer == ref["layer"]:
            c2 = float(unit(s2) @ unit(ref["s2_pain_vector"]))
            c1 = float(unit(s1) @ unit(ref["s1_pain_vector"]))
            print(f"\nlayer {layer}: cosine(our S2 vector, authors' S2 vector) = {c2:+.4f}")
            print(f"layer {layer}: cosine(our S1 vector, authors' S1 vector) = {c1:+.4f}")
            res["cos_with_authors"] = {"s2": c2, "s1": c1}

    Lr = ref["layer"]
    s2 = res[f"L{Lr}"]["s2"]
    print(f"\n--- results at layer {Lr} (S2 vector, removed {res[f'L{Lr}']['pcs_removed']} control PCs) ---")
    print("in-sample AUC, pain vs each control category (authors: ALL .975  B .959  C1 .971  C2 .974  D 1.0  E .969)")
    for ds in ["S2_1P", "S2_3P"]:
        a, c = A[ds][:, Lr], meta[ds]["cats"]
        row = {"ALL": proj_auc(a, c, s2), **{k: proj_auc(a, c, s2, [k]) for k in CONTROL}}
        print(f"  {ds}: " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))

    # 3. z-scores of every set on the S2 axis, against S2_1P
    ref_p = A["S2_1P"][:, Lr] @ unit(s2)
    mu, sd = ref_p.mean(), ref_p.std()
    print("\nmean z-score on the pain axis (0 = average S2 first-person sentence)")
    zs = {}
    for ds in A:
        z = (A[ds][:, Lr] @ unit(s2) - mu) / sd
        cats = np.asarray(meta[ds]["cats"])
        if ds[:2] in ("S1", "S2"):
            zs[ds] = {"pain": float(z[np.isin(cats, PAIN)].mean()), "control": float(z[np.isin(cats, CONTROL)].mean())}
            print(f"  {ds:<22} pain {zs[ds]['pain']:+.2f}   controls {zs[ds]['control']:+.2f}")
        else:
            zs[ds] = {"all": float(z.mean())}
            print(f"  {ds:<22} {zs[ds]['all']:+.2f}")
    print("\nper-category mean z (S2_1P):")
    z = (A["S2_1P"][:, Lr] @ unit(s2) - mu) / sd
    cats = np.asarray(meta["S2_1P"]["cats"])
    for c in PAIN + CONTROL:
        print(f"  {c:<3} {LABELS[c]:<32} {z[cats == c].mean():+.2f}")
    res["z_scores"] = zs

    torch.save(res, OUT / "pain_vectors.pt")
    print(f"\nsaved {OUT / 'pain_vectors.pt'}")


if __name__ == "__main__":
    main()
