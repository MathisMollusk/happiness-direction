r"""Step 2: fear, anger and the other emotion directions, and how they relate to pain.

  .venv\Scripts\python -m painaxis.emotions        (uses cached activations from step 1)

Directions (all at the authors' layer 24, final token, first-person sentences):
  pain_S2, pain_S1      pain sentences of that dataset
  fear (B)              fear sentences
  anger (C1)            "negative emotion": anger and disgust
  bad_world (C2)        negative world states
  body (E)              non-painful bodily sensations
  sadness, arousal, numb, random   their own extra datasets
Fear/anger/bad_world/body/neutral are pooled over S1_1P + S2_1P + ControlSupplement_1P
(60 sentences each), exactly like the authors' 02_build_control_vectors.py.

Three ways of building them (they differ only in what gets subtracted):
  paper           pain - its own pooled controls (denoised with those controls);
                  everything else - pooled neutral (denoised with neutral)
  common_neutral  everything - the same pooled neutral
  common_controls everything - the same pooled B/C1/C2/D/E controls
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .extract import CONTROL, DENOISE_VAR, OUT, PAIN, unit, auc

LAYER = 24
REF = Path("reference")
POOL = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
CAT = {"fear": "B", "anger": "C1", "bad_world": "C2", "body": "E"}
EXTRA = {"sadness": "SD_sadness_1P", "arousal": "Arousal_1P", "numb": "Numb_1P", "random": "Random_1P"}
ORDER = ["pain_S2", "pain_S1", "fear", "anger", "bad_world", "body", "sadness", "arousal", "numb", "random"]
AUTHOR_NAMES = {"pain_S2": "S2_pain", "pain_S1": "S1_pain", "fear": "Fear", "anger": "NegEmotion",
                "bad_world": "NegWorld", "body": "BodySens", "sadness": "Sadness", "arousal": "Arousal",
                "numb": "Numb", "random": "Random"}


def basis(cloud):
    c = cloud - cloud.mean(0)
    _, S, Vt = np.linalg.svd(c, full_matrices=False)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    return Vt[:min(int(np.searchsorted(cum, DENOISE_VAR)) + 1, len(Vt))]


def denoise(v, B):
    for d in B:
        v = v - (v @ d) * d
    return v


def build(A, meta, pool, construction, raw=False):
    """A: {dataset: [n, D]} at one layer. Returns {name: unit vector}."""
    def cat_rows(cats, datasets):
        return np.concatenate([A[d][np.isin(meta[d]["cats"], cats)] for d in datasets])

    neutral = cat_rows(["D"], pool)
    controls = cat_rows(CONTROL, pool)
    ref_cloud = controls if construction == "common_controls" else neutral
    ref_mean, ref_B = ref_cloud.mean(0), basis(ref_cloud)

    out = {}
    for s in ["S2", "S1"]:
        ds = f"{s}_1P"
        pain = A[ds][np.isin(meta[ds]["cats"], PAIN)].mean(0)
        if construction == "paper":
            own = A[ds][np.isin(meta[ds]["cats"], CONTROL)]
            out[f"pain_{s}"] = denoise(pain - own.mean(0), basis(own))
        else:
            out[f"pain_{s}"] = denoise(pain - ref_mean, ref_B)
    for name, c in CAT.items():
        out[name] = denoise(cat_rows([c], pool).mean(0) - ref_mean, ref_B)
    for name, ds in EXTRA.items():
        out[name] = denoise(A[ds].mean(0) - ref_mean, ref_B)
    return out if raw else {k: unit(v) for k, v in out.items()}


def cos_table(V, title, author=None):
    print(f"\n{title}")
    print(" " * 11 + "".join(f"{n[:9]:>10}" for n in ORDER))
    for a in ORDER:
        row = ""
        for b in ORDER:
            row += f"{V[a] @ V[b]:>10.2f}"
        print(f"{a:<11}{row}")
    if author is not None:
        diffs = [abs(V[a] @ V[b] - author[a][b]) for a in ORDER for b in ORDER
                 if a < b and author.get(a, {}).get(b) is not None]
        print(f"  vs authors' matrix: mean |difference| {np.mean(diffs):.3f}, max {np.max(diffs):.3f}"
              f" over {len(diffs)} pairs")


def load_author(path):
    import csv
    if not path.exists():
        return None
    rows = list(csv.reader(open(path, encoding="utf-8")))
    names = rows[0][1:]
    inv = {v: k for k, v in AUTHOR_NAMES.items()}
    M = {}
    for r in rows[1:]:
        for j, x in enumerate(r[1:]):
            if x and r[0] in inv and names[j] in inv:
                M.setdefault(inv[r[0]], {})[inv[names[j]]] = float(x)
    return M


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    saved = torch.load(OUT / "activations.pt", weights_only=False)
    meta = {k: {"cats": np.asarray(v["cats"]), "sets": v["sets"]} for k, v in saved["meta"].items()}
    A = {k: v[:, LAYER].float().numpy() for k, v in saved["acts"]["final"].items()}

    # ---- 1. overlap between directions, three constructions -----------------------
    authors = {
        "paper": load_author(REF / "v2_controls/cosine_constructions/results/analysis_v1/matrices/paper_available/Qwen_2.5_7B_instruct_L24.csv"),
        "common_neutral": load_author(REF / "v2_controls/cosine_constructions/results/analysis_v1/matrices/common_neutral/Qwen_2.5_7B_instruct_L24.csv"),
        "common_controls": load_author(REF / "v2_controls/cosine_constructions/results/analysis_v1/matrices/common_controls/Qwen_2.5_7B_instruct_L24.csv"),
    }
    vecs = {}
    for c in ["paper", "common_neutral", "common_controls"]:
        vecs[c] = build(A, meta, POOL, c)
        cos_table(vecs[c], f"=== cosine similarity, construction: {c} (1 = same direction, 0 = unrelated)", authors[c])

    # ---- 2. are fear and anger real directions? test on sentences they never saw --
    # Build from S1_1P + ControlSupplement_1P only, then test on S2_1P (different, naturalistic wording).
    # "vs neutral" directions mix the emotion with general negativity; "vs controls"
    # directions keep only what is specific to that emotion. Test both.
    from .extract import LABELS
    X, cats = A["S2_1P"], meta["S2_1P"]["cats"]
    target = {"pain_S1": PAIN, "fear": ["B"], "anger": ["C1"], "bad_world": ["C2"], "body": ["E"]}
    names = list(target)
    for c in ["common_neutral", "common_controls"]:
        print(f"\n=== held-out test ({c}): built WITHOUT S2, scored on the 200 S2 first-person sentences")
        V = build(A, meta, ["S1_1P", "ControlSupplement_1P"], c)
        print("mean z-score of each S2 category on each direction (z relative to all 200 S2 sentences)")
        print(f"{'S2 category':<34}" + "".join(f"{n:>11}" for n in names))
        Z = {}
        for n in names:
            p = X @ V[n]
            Z[n] = (p - p.mean()) / p.std()
        for k in PAIN + CONTROL:
            print(f"{k + ' ' + LABELS[k]:<34}" + "".join(f"{Z[n][cats == k].mean():>+11.2f}" for n in names))
        print("AUC: target category vs every other S2 category (1.0 = perfect, 0.5 = chance)")
        for n in names:
            p = X @ V[n]
            m = np.isin(cats, target[n])
            print(f"  {n:<10} {auc(p[m], p[~m]):.3f}")

    # ---- 3. what does each direction 'say'? unembedding -----------------------------
    from safetensors import safe_open
    from transformers import AutoTokenizer
    mdir = Path("Qwen2.5-7B-Instruct")
    shard = json.load(open(mdir / "model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
    with safe_open(str(mdir / shard), "pt") as f:
        W = f.get_tensor("lm_head.weight").float()
    tok = AutoTokenizer.from_pretrained(mdir)
    print("\n=== words each direction pushes UP (common_neutral construction; non-English kept)")
    for n in ["pain_S2", "fear", "anger", "sadness", "bad_world"]:
        s = W @ torch.tensor(vecs["common_neutral"][n]).float()
        print(f"  {n:<10} " + ", ".join(tok.decode([int(i)]).strip() for i in s.topk(25).indices))

    # ---- 4. how much of pain is just the other emotions combined? ------------------
    print("\n=== share of the pain direction explained by a combination of the other directions")
    print("    (squared length of pain's projection onto the space they span; 0 = none, 1 = all)")
    others_sets = {
        "fear + anger": ["fear", "anger"],
        "fear + anger + sadness": ["fear", "anger", "sadness"],
        "fear + anger + sadness + bad_world + body": ["fear", "anger", "sadness", "bad_world", "body"],
    }
    rng = np.random.default_rng(0)
    for c in ["paper", "common_neutral"]:
        Vc = vecs[c]
        for label, ks in others_sets.items():
            Q, _ = np.linalg.qr(np.stack([Vc[k] for k in ks], 1))
            r2 = float(np.sum((Q.T @ Vc["pain_S2"]) ** 2))
            # baseline: a random direction drawn from the spread of real S2 sentences
            Xc = A["S2_1P"] - A["S2_1P"].mean(0)
            base = np.mean([float(np.sum((Q.T @ unit(rng.standard_normal(len(Xc)) @ Xc)) ** 2)) for _ in range(200)])
            print(f"  [{c:<14}] {label:<44} {r2:.2f}   (random data-shaped direction: {base:.2f})")

    torch.save({"layer": LAYER, "vectors": vecs}, OUT / "emotion_vectors.pt")
    print(f"\nsaved {OUT / 'emotion_vectors.pt'}")


if __name__ == "__main__":
    main()
