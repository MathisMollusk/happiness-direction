r"""Diagnose generation memory: dynamic vs static KV cache, batch 8, 512 new tokens.

  .venv\Scripts\python -m painaxis.memtest
"""

import time

import torch

from . import runlog
from .extract import load_model


def main():
    runlog.start("memtest")
    model, tok = load_model("Qwen2.5-7B-Instruct")
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": "List animals, one per line. Keep going as long as you like."}],
                                     tokenize=False, add_generation_prompt=True)] * 8
    enc = tok(texts, return_tensors="pt", padding=True).to(model.input_device)
    base = torch.cuda.memory_allocated() / 2**30
    print(f"model allocated {base:.2f} GiB")
    for cache in ["dynamic", "static"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        kw = {"cache_implementation": "static", "disable_compile": True} if cache == "static" else {}
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=512, min_new_tokens=512, do_sample=True, temperature=0.7,
                               pad_token_id=tok.pad_token_id, **kw)
        print(f"  {cache:<8} {time.time() - t0:5.1f}s  peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB  "
              f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB  ({o.shape[1]} tokens)", flush=True)


if __name__ == "__main__":
    main()
