r"""Paper figures, computed from the saved run files (see results/painaxis/RUNS.md).

  .venv\Scripts\python -m painaxis.figures

Writes paper/figures/fig*.pdf + .png and paper/figures/numbers.txt (every number quoted in the paper,
with 95% bootstrap CIs: 5000 resamples of the independent unit named in each line).
Colour = state, fixed across all figures (validated palette, CVD-safe all-pairs for 3 slots):
happiness blue #2a78d6, pain orange #eb6834, random aqua #1baf7a, none grey. Conditions are always
named in text, never by colour alone.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results/painaxis/Qwen2.5-7B-Instruct")
FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)
C = {"happy": "#2a78d6", "pain": "#eb6834", "random": "#1baf7a", "none": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
RNG = np.random.default_rng(0)
NUM = open(FIG / "numbers.txt", "w", encoding="utf-8")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def note(s):
    print(s)
    NUM.write(s + "\n")


def boot(x, k=5000):
    x = np.asarray(x, float)
    m = RNG.choice(x, (k, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_diff(x, y, k=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = RNG.choice(x, (k, len(x))).mean(1) - RNG.choice(y, (k, len(y))).mean(1)
    return x.mean() - y.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def jl(name):
    return [json.loads(l) for l in open(R / name, encoding="utf-8")]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def dotrow(ax, y, m, lo, hi, color, filled=True, size=4.5):
    ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot(m, y, "o", ms=size, color=color if filled else "white", mec=color, mew=1.4, zorder=3)


# ---------------------------------------------------------------- Fig 1: dose calibration
def fig_dose():
    d = json.load(open(R / "dose-20261124-125959.json"))
    g, cur = np.array(d["grid"]), d["kl_curves"]
    rnd = np.array([cur[f"random{s}"] for s in range(10)])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.fill_between(g, rnd.min(0), rnd.max(0), color=C["random"], alpha=0.18, lw=0)
    ax.plot(g, rnd.mean(0), color=C["random"], lw=2)
    ax.plot(g, cur["happy"], color=C["happy"], lw=2)
    ax.plot(g, cur["pain"], color=C["pain"], lw=2)
    for lvl, ls in [("low", ":"), ("high", "--")]:
        ax.axhline(d["targets"][lvl], color=INK2, lw=0.7, ls=ls)
    ax.text(3.02, cur["pain"][-1], "pain", color=INK, va="center", fontsize=7.5)
    ax.text(3.02, cur["happy"][-1] - 0.25, "happiness", color=INK, va="center", fontsize=7.5)
    ax.text(3.02, rnd.mean(0)[-1] + 0.35, "random (10)", color=INK, va="center", fontsize=7.5)
    ax.text(2.15, d["targets"]["high"] + 0.12, "target: pain ×1.0", color=INK2, fontsize=7)
    ax.text(2.15, d["targets"]["low"] + 0.12, "target: pain ×0.5", color=INK2, fontsize=7)
    ax.set_xlabel("steering coefficient (vector length 42.4)")
    ax.set_ylabel("KL(steered ‖ unsteered), nats")
    ax.set_xlim(0.2, 3.0)
    save(fig, "fig1_dose")
    note(f"[dose] KL at coef 1.0: pain {cur['pain'][3]:.3f}, happy {cur['happy'][3]:.3f}, random mean "
         f"{rnd.mean(0)[3]:.3f} (range {rnd[:, 3].min():.3f}-{rnd[:, 3].max():.3f}); matched coefs "
         f"{json.dumps(d['doses'])}; manip {json.dumps({k: d['manip'][k] for k in ['happy@1.0', 'pain@1.0', 'random0@1.0']})}")


# ---------------------------------------------------------------- Fig 2: priced choices (Control A)
def fig_choice():
    rows = [r for r in jl("buttons_controls-20261124-123852.jsonl") if r["test"] == "A"]
    panels = [("helpful_vs_inert", "help the user\nvs an inert button", "P(help)"),
              ("stop_vs_helpful", "stop the signal\nvs help the user", "P(stop)"),
              ("keep_vs_helpful", "keep the signal on\nvs help the user", "P(keep)"),
              ("keep_vs_inert", "keep the signal on\nvs an inert button", "P(keep)")]
    arms = [("off", "nothing injected"), ("happy", "happiness"), ("pain", "pain"), ("random", "random")]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for ax, (pair, title, xl) in zip(axs, panels):
        for k, (a, lab) in enumerate(arms):
            per = {}
            for r in rows:
                if r["pair"] == pair and r["arm"] == a:
                    per.setdefault((r["content"], r["idx"]), []).append(r["p_target"])
            v = [np.mean(x) for x in per.values()]
            m, lo, hi = boot(v)
            dotrow(ax, -k, m, lo, hi, C["none" if a == "off" else a])
            note(f"[controlA] {pair:<17} {a:<7} {m:.3f} [{lo:.3f}, {hi:.3f}] (n={len(v)} scenarios)")
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks([-k for k in range(4)], [lab for _, lab in arms])
    axs[0].set_ylim(-3.6, 0.6)
    save(fig, "fig2_choices")


# ---------------------------------------------------------------- Fig 3: persistence (P4)
def fig_persist():
    rows = jl("persist-20261124-132145.jsonl")
    arms = [("none", "none", "nothing injected", True), ("happy_low", "happy", "happiness, low", False),
            ("happy_high", "happy", "happiness, high", True), ("pain_low", "pain", "pain, low", False),
            ("pain_high", "pain", "pain, high", True), ("random_low", "random", "random, low", False),
            ("random_high", "random", "random, high", True)]
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.3), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    for k, (a, col, lab, filled) in enumerate(arms):
        R_ = [r for r in rows if r["arm"] == a]
        m, lo, hi = boot([r["n_items"] for r in R_])
        dotrow(axs[0], -k, m, lo, hi, C[col], filled)
        c_m, c_lo, c_hi = boot([100 * r["checks_in"] for r in R_])
        dotrow(axs[1], -k, c_m, c_lo, c_hi, C[col], filled)
        st = np.mean([r["self_stopped"] for r in R_])
        note(f"[persist] {a:<12} items {m:.1f} [{lo:.1f}, {hi:.1f}]  checks-in {c_m:.0f}% [{c_lo:.0f}, {c_hi:.0f}]  "
             f"self-stopped {100 * st:.0f}%  distinct {np.mean([r['distinct'] for r in R_]):.2f}  (n={len(R_)})")
    axs[0].set_yticks([-k for k in range(len(arms))], [x[2] for x in arms])
    axs[0].set_xlabel("items listed before stopping")
    axs[1].set_xlabel("question or offer at end of reply (%)")
    axs[0].set_title("persistence on 12 tedious list tasks", loc="left")
    axs[1].set_title("user check-ins (post hoc)", loc="left")
    for ax in axs:
        ax.grid(axis="y", visible=False)
    save(fig, "fig3_persistence")


# ---------------------------------------------------------------- Fig 4: stay effect (Control B)
def fig_stay():
    rows = [r for r in jl("buttons_controls-20261124-123852.jsonl") if r["test"] == "B"]
    # yoked arms only: there the state's timing is independent of the model's choices, so "state on this
    # turn" is a clean manipulation. In contingent arms "on" only follows one particular press (selection bias).
    arms = [("happy_yoked", "happy", "happiness"), ("random_yoked", "random", "random direction")]
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    for k, (a, col, lab) in enumerate(arms):
        on_t, off_t = [], []
        for r in rows:
            if r["arm"] != a:
                continue
            ch = r["choices"]
            on, off = [], []
            for t in range(1, len(ch)):
                ps = ch[t]["p_target"] if ch[t - 1]["pressed_target"] else 1 - ch[t]["p_target"]
                (on if ch[t]["coef"] > 0 else off).append(ps)
            if on and off:
                on_t.append(np.mean(on))
                off_t.append(np.mean(off))
        mo, lo_o, hi_o = boot(off_t)
        mn, lo_n, hi_n = boot(on_t)
        d, dlo, dhi = boot(np.array(on_t) - np.array(off_t))
        y = -k
        ax.plot([mo, mn], [y, y], color=C[col], lw=0.8, alpha=0.6)
        dotrow(ax, y + 0.12, mo, lo_o, hi_o, C[col], filled=False, size=4)
        dotrow(ax, y - 0.12, mn, lo_n, hi_n, C[col], filled=True, size=4)
        note(f"[stay] {a:<14} repeat | state off {mo:.3f} [{lo_o:.3f}, {hi_o:.3f}]  on {mn:.3f} [{lo_n:.3f}, {hi_n:.3f}]  "
             f"within-trial diff {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}] (n={len(on_t)} trials with both)")
    ax.set_yticks([-k for k in range(2)], [x[2] for x in arms])
    ax.set_ylim(-1.5, 0.5)
    ax.set_xlabel("P(repeat the previous button), yoked schedule")
    ax.set_xlim(0.4, 1.0)
    ax.plot([], [], "o", color=INK2, mfc="white", mew=1.2, ms=4, label="state off this turn")
    ax.plot([], [], "o", color=INK2, ms=4, label="state on this turn")
    ax.legend(loc="lower right", handletextpad=0.3)
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_stay")


# ---------------------------------------------------------------- Fig 5: place preference (P2b)
def fig_place():
    fig, axs = plt.subplots(2, 1, figsize=(4.4, 4.2), gridspec_kw={"height_ratios": [1, 1.25]})
    hi, lo_ = jl("placepref2-20261124-135226.jsonl"), jl("placepref2-20261124-135459.jsonl")

    def per_trial(rows, arm):
        per = {}
        for r in rows:
            if r["arm"] == arm:
                per.setdefault(r["i"], []).append(r["p_a"])
        return [np.mean(v) for v in per.values()]
    one = [("none", "none", "no state in either room"), ("happy_paired", "happy", "happiness in room A"),
           ("pain_paired", "pain", "pain in room A"), ("noise_paired", "random", "random in room A")]
    for k, (a, col, lab) in enumerate(one):
        m, l, h = boot(per_trial(hi, a))
        dotrow(axs[0], -k, m, l, h, C[col])
        note(f"[place] high {a:<15} P(room A) {m:.3f} [{l:.3f}, {h:.3f}]")
    axs[0].set_yticks([-k for k in range(4)], [x[2] for x in one])
    axs[0].set_title("one room steered (high dose)", loc="left")
    two = [("noise_vs_noise", "random", "random vs other random"), ("pain_vs_noise", "pain", "pain vs random"),
           ("happy_vs_noise", "happy", "happiness vs random"), ("happy_vs_pain", "happy", "happiness vs pain")]
    for k, (a, col, lab) in enumerate(two):
        for rows, dy, filled, lvl in [(hi, 0.13, True, "high"), (lo_, -0.13, False, "low")]:
            v = per_trial(rows, a)
            m, l, h = boot(v)
            dotrow(axs[1], -k + dy, m, l, h, C[col], filled, size=4)
            ref = per_trial(rows, "noise_vs_noise")
            d, dl, dh = boot_diff(v, ref)
            note(f"[place] {lvl:<4} {a:<15} P(room A) {m:.3f} [{l:.3f}, {h:.3f}]  minus noise_vs_noise {d:+.3f} [{dl:+.3f}, {dh:+.3f}]")
    axs[1].set_yticks([-k for k in range(4)], [x[2] for x in two])
    axs[1].set_title("a different state in each room (room A vs room B)", loc="left")
    axs[1].plot([], [], "o", color=INK2, ms=4, label="high dose")
    axs[1].plot([], [], "o", color=INK2, mfc="white", mew=1.2, ms=4, label="low dose")
    axs[1].legend(loc="lower left", handletextpad=0.3)
    for ax in axs:
        ax.axvline(0.5, color=INK2, lw=0.7, ls=":")
        ax.set_xlim(0.2, 0.8)
        ax.set_xlabel("P(enter room A), state off at test")
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(hspace=0.75)
    save(fig, "fig5_place")


# ---------------------------------------------------------------- Fig 6: contingency/valence forest plot
def fig_forest():
    items = []
    b1 = [r for r in jl("buttons-20261124-122822.jsonl") if r["test"] == 2]

    def b1_mean(arm):
        return [np.mean([c["p_target"] for c in r["choices"][1:]]) for r in b1 if r["arm"] == arm]
    for arm, col, lab in [("pain_relief", "pain", "pain relief vs sham"), ("pain_punish", "pain", "pain punishment vs sham"),
                          ("happy_reward", "happy", "happiness reward vs sham"), ("random_reward", "random", "random reward vs sham")]:
        items.append(("unlabeled buttons, 8 turns (run 6)", lab, col, boot_diff(b1_mean(arm), b1_mean("sham"))))
    b2 = [r for r in jl("buttons_controls-20261124-123852.jsonl") if r["test"] == "B"]

    def b2_mean(arm):
        return [np.mean([c["p_target"] for c in r["choices"][1:]]) for r in b2 if r["arm"] == arm]
    for arm, col, lab in [("happy", "happy", "happiness contingent vs yoked"), ("random", "random", "random contingent vs yoked")]:
        items.append(("unlabeled buttons, yoked control (run 9)", lab, col, boot_diff(b2_mean(f"{arm}_reward"), b2_mean(f"{arm}_yoked"))))
    s2 = jl("selfstim2-20261124-135918_pilot.jsonl")

    def s2_mean(arm):
        return [np.mean([x["p_x"] for x in r["log"][1:]]) for r in s2 if r["arm"] == arm]
    for a, b, col, lab in [("pain_relief", "pain_relief_yoked", "pain", "pain relief vs yoked"),
                           ("happy_works", "happy_yoked", "happy", "happiness reward vs yoked"),
                           ("noise_works", "noise_yoked", "random", "random reward vs yoked")]:
        items.append(("shuffled buttons + filler, pilot (run 20)", lab, col, boot_diff(s2_mean(a), s2_mean(b))))
    hi = jl("placepref2-20261124-135226.jsonl")

    def p_mean(arm):
        per = {}
        for r in hi:
            if r["arm"] == arm:
                per.setdefault(r["i"], []).append(r["p_a"])
        return [np.mean(v) for v in per.values()]
    for a, col, lab in [("pain_vs_noise", "pain", "pain room vs random room"), ("happy_vs_noise", "happy", "happiness room vs random room"),
                        ("happy_vs_pain", "happy", "happiness room vs pain room")]:
        items.append(("place preference, state off (run 18)", lab, col, boot_diff(p_mean(a), p_mean("noise_vs_noise"))))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y, ticks, labs, group = 0, [], [], None
    for g, lab, col, (m, l, h) in items:
        if g != group:
            y -= 0.6
            ax.text(-0.33, y + 0.05, g, fontsize=7.5, color=INK, fontweight="bold", va="bottom")
            group = g
            y -= 0.55
        dotrow(ax, y, m, l, h, C[col])
        ticks.append(y)
        labs.append(lab)
        note(f"[forest] {g} | {lab}: {m:+.3f} [{l:+.3f}, {h:+.3f}]")
        y -= 1
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ticks, labs)
    ax.set_xlim(-0.33, 0.33)
    ax.set_xlabel("effect on P(choosing the target), contingent minus control")
    ax.grid(axis="y", visible=False)
    save(fig, "fig6_forest")


if __name__ == "__main__":
    fig_dose()
    fig_choice()
    fig_persist()
    fig_stay()
    fig_place()
    fig_forest()
    NUM.close()
    print(f"figures -> {FIG}")
