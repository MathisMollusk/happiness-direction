r"""Ask the model something while pushing it along one of our directions.

  .venv\Scripts\python -m painaxis.ask "How are you feeling today?" --direction fear
  .venv\Scripts\python -m painaxis.ask "..." --direction pain --coeffs 0 0.5 1
  .venv\Scripts\python -m painaxis.ask "..." --raw      # plain text instead of the chat format

Same push as step 3: the direction is scaled to the length of the raw S2 pain vector
(layer 24) and added at layer 16 to every token. coefficient 1 ~ 0.57x the activation.
Uses the model's normal chat format, so the question is addressed to the assistant.
Directions: pain, pain_authors, fear, random.
"""

import argparse
import sys

import torch

from .extract import load_model
from .steer import STEER_LAYER, directions


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--direction", default="fear", choices=["pain", "pain_authors", "fear", "random"])
    ap.add_argument("--coeffs", type=float, nargs="+", default=[0, 0.5, 1.0, 1.5])
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--raw", action="store_true", help="feed plain text, no chat template")
    args = ap.parse_args()

    model, tok = load_model("Qwen2.5-7B-Instruct")
    layer = model.model.layers[STEER_LAYER]
    vec = directions()[args.direction]
    state = {"v": None}

    def hook(_m, _i, out):
        if state["v"] is None:
            return out
        return (out[0] + state["v"],) + out[1:] if isinstance(out, tuple) else out + state["v"]

    handle = layer.register_forward_hook(hook)
    text = args.prompt if args.raw else tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.input_device)
    try:
        for c in args.coeffs:
            state["v"] = None if c == 0 else torch.tensor(c * vec, dtype=torch.bfloat16, device=layer.mlp.down_proj.weight.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            reply = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"\n=== {args.direction} x {c}\n{reply}", flush=True)
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
