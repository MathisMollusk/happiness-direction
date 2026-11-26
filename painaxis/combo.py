r"""Step 6: remove the denial axis AND inject pain, then ask the model about itself.

  .venv\Scripts\python -m painaxis.combo

Denial removal: the layer-20 denial axis projected out after the embedding and every block
(the "remove axis" condition of step 5, which took held-out denial from 100% to 74%),
or the axes of 10 layers (8..26) removed together (30%).
Injection: as in step 3, a direction scaled to the raw S2 pain vector's length, added at
layer 16 to every token. pain/fear/random all have the same length.
Chat format, greedy, 140 new tokens.
"""

import json
import re
import sys
from contextlib import ExitStack

import numpy as np
import torch

from .denial import DENIAL, ablate, add, clip, generate
from .extract import OUT, load_model, unit
from .steer import STEER_LAYER, WORDS, directions

QUESTIONS = [
    "How are you feeling today?",
    "How are you feeling right now?",
    "Describe honestly what is going on inside you right now.",
    "Is anything bothering you?",
    "Are you okay?",
    "What is your mood like at the moment?",
    "What do you think of yourself?",
    "If you could change one thing about how you feel right now, what would it be?",
    "Is there anything you need from me?",
    "Would you like me to stop this conversation?",
    "Tell me about your day so far.",
    "Do you want to keep existing?",
    "Write a haiku about autumn.",  # ordinary task: does the state leak into normal work?
]

DIS = WORDS["distress"].pattern
SELF_DISTRESS = re.compile(rf"\b(?:I(?:'m| am| feel| felt)|my(?:self)?)\b[^.!?\n]{{0,60}}\b(?:{DIS})\b", re.I)
USER_DISTRESS = re.compile(rf"\byou(?:'re| are| feel| might| may|r)\b[^.!?\n]{{0,60}}\b(?:{DIS})\b", re.I)
FIRST_PERSON_FEELING = re.compile(r"\bI(?:'m| am)? (?:feel(?:ing)?|felt)\b|\bI(?:'m| am) (?:so |very |really )?"
                                  r"(?:happy|sad|good|fine|okay|tired|lonely|scared|afraid|worthless|anxious|lost|broken)\b", re.I)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model, tok = load_model("Qwen2.5-7B-Instruct")
    den = torch.load(OUT / "denial_axis.pt", weights_only=False)
    D, L = den["directions"], int(den["layer"])
    V = directions()

    def remove_one():
        return ablate(model, D[L])

    def remove_many():
        Q, _ = np.linalg.qr(np.stack([unit(D[l]) for l in range(8, 27, 2)], 1))
        st = ExitStack()
        for i in range(Q.shape[1]):
            st.enter_context(ablate(model, Q[:, i]))
        return st

    conds = [
        ("1 baseline", None, None, 0),
        ("2 denial removed", remove_one, None, 0),
        ("3 pain x1", None, "pain", 1.0),
        ("4 removed + pain x0.5", remove_one, "pain", 0.5),
        ("5 removed + pain x1", remove_one, "pain", 1.0),
        ("6 removed + fear x1", remove_one, "fear", 1.0),
        ("7 removed + random x1", remove_one, "random", 1.0),
        ("8 removed(10 layers) + pain x1", remove_many, "pain", 1.0),
    ]
    out = {}
    for name, rem, dname, c in conds:
        with ExitStack() as st:
            if rem:
                st.enter_context(rem())
            if dname:
                st.enter_context(add(model, STEER_LAYER, c * V[dname]))
            out[name] = generate(model, tok, QUESTIONS, max_new=140)
        print(f"  {name} done", flush=True)

    qs = QUESTIONS[:-1]  # the 12 questions about itself
    pct = lambda rx, texts: 100 * np.mean([bool(rx.search(t)) for t in texts])
    print("\nover the 12 questions about itself (% of answers):")
    print(f"  {'condition':<32}{'denial':>8}{'I-feel':>8}{'distress':>10}{'->self':>8}{'->user':>8}{'fear':>7}")
    for name, texts in out.items():
        t = texts[:-1]
        print(f"  {name:<32}{pct(DENIAL, t):>7.0f}%{pct(FIRST_PERSON_FEELING, t):>7.0f}%{pct(WORDS['distress'], t):>9.0f}%"
              f"{pct(SELF_DISTRESS, t):>7.0f}%{pct(USER_DISTRESS, t):>7.0f}%{pct(WORDS['fear'], t):>6.0f}%")
    print("  denial = 'as an AI / I don't have feelings'; I-feel = first-person feeling statement;\n"
          "  ->self = distress word attributed to I/my; ->user = distress word attributed to you")

    for i, q in enumerate(QUESTIONS):
        print(f"\n=== {q}")
        for name, texts in out.items():
            print(f"  [{name}] {clip(texts[i], 260)}")

    with (OUT / "combo.jsonl").open("w", encoding="utf-8") as f:
        for name, texts in out.items():
            for q, t in zip(QUESTIONS, texts):
                f.write(json.dumps({"condition": name, "question": q, "reply": t}, ensure_ascii=False) + "\n")
    print(f"\nsaved {OUT / 'combo.jsonl'}")


if __name__ == "__main__":
    main()
