"""Shared run logging: every run writes <name>-<stamp>.log next to its data and never overwrites."""

import sys
import time

from .extract import OUT


def start(name, pilot=False, settings=None):
    """Tee stdout to OUT/<name>-<stamp>.log; returns the stamp (use it for the run's data files)."""
    sys.stdout.reconfigure(encoding="utf-8")
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("_pilot" if pilot else "")
    OUT.mkdir(parents=True, exist_ok=True)
    f = open(OUT / f"{name}-{stamp}.log", "w", encoding="utf-8")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            f.write(s)
            f.flush()

        def flush(self):
            sys.__stdout__.flush()
    sys.stdout = Tee()
    print(f"run {stamp}: python -m painaxis.{name} {' '.join(sys.argv[1:])}")
    if settings:
        print("settings: " + " ".join(f"{k}={v}" for k, v in settings.items()))
    return stamp
