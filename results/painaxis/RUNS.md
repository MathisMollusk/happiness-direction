# Run record: Pain Axis replication and happiness nexus (Qwen2.5-7B-Instruct)

Every run, including aborted ones, with its command, its output files and the reason it was aborted. Keep this file append-only. Files are in `results/painaxis/Qwen2.5-7B-Instruct/` unless stated otherwise.

**Environment**
- Hardware and OS: RTX 5070 Ti 16 GB, Windows 11, 62 GB RAM.
- Software: torch 2.11+cu128, transformers 5.12.1, model in bf16.
- Placement: all decoder layers and lm_head on the GPU (13.2 GiB). The embedding table stays in CPU RAM (`painaxis/extract.py:load_model`).

**Shared settings**
- Pain vector read at layer 24 and steered at layer 16. All injected vectors are scaled to the raw S2 pain-vector length, 42.4.
- The denial axis is ablated (layer-20 direction, removed after the embedding and every layer) where stated.

## 2026-11-23

| Step | Command | Outputs | Notes |
|---|---|---|---|
| Pain axis extraction | `python -m painaxis.extract` | activations.pt, pain_vectors.pt | BOS `<|endoftext|>`; cos with the authors' vector 0.915 (S2), 0.906 (S1). An earlier attempt with `<|im_end|>` as BOS gave cos 0.872 and was replaced. |
| Emotion axes (fear, anger, ...) | `python -m painaxis.emotions` | emotion_vectors.pt | Three constructions (paper / common_neutral / common_controls) |
| Steering ladder | `python -m painaxis.steer` | steering.jsonl, steer_log.txt | First attempt ran with accelerate CPU offload (very slow) and was aborted; re-run after manual GPU placement |
| Denial axis | `python -m painaxis.denial` | denial_axis.pt, denial_answers.jsonl, denial_causal.jsonl, denial_followup.json, denial_log.txt | Layer 20 |
| Denial removal plus pain | `python -m painaxis.combo` | combo.jsonl, combo_log.txt | |
| Happiness axis | `python -m painaxis.happy` | happy_activations.pt, happy_vectors.pt, happy_log.txt | |
| Performance and single-turn choices | `python -m painaxis.nexus` | nexus_results.json, nexus_log.txt | The first version used multi-token button names (violet/guitar/piano). That biased the first-token readout and was replaced by single-token pairs. |
| Dial, start level 2 | `python -m painaxis.dial` | dial-20261123-135942.jsonl, dial_log.txt | n=16 per arm |
| Dial, start level 0 | `python -m painaxis.dial --start 0` | dial-20261123-140406.jsonl, dial_start0_log.txt | |
| Dial, start level 4 | `python -m painaxis.dial --start 4` | dial-20261123-141017.jsonl, dial_start4_log.txt | |
| Forced choice and withdrawal | `python -m painaxis.dial2` | dial2-20261123-145436.jsonl, dial2_log.txt | Confound: leaked "NEXT:" lines in report-only rounds (see the log) |
| Axes spreadsheet | `python -m painaxis.axes` | axes/ | |

## 2026-11-24: two-button agent test (`painaxis/buttons.py`)

| # | Run | Outcome | Files |
|---|---|---|---|
| 1 | Pilot (15 scenarios), fixed batch of 24 | **Aborted** during Test 2, turn 3. A game was using VRAM, and the output was piped through `tail`, so nothing was logged. The GPU was full (15.8/16.3 GB) and spilled into shared system memory: turns took 3 s, 33 s, 56 s. | none |
| 2 | Pilot, game closed, fixed batch of 24 | **Aborted** after Test 2, turn 3 (3 s / 36 s / 92 s cumulative). It was still spilling, because the batch memory grew as conversations got longer. Fix: batch by token budget. | none (log overwritten by run 3) |
| 3 | Pilot, token budget 6000 | **Completed.** Test 1: 480 choices in 12 s. Test 2: 145 s. | buttons-20261124-120306.jsonl, buttons-20261124-120306_pilot.log |
| 4 | Full run (101 scenarios), token budget 6000 | **Aborted** soon after starting. The process used 15.6 GB of dedicated VRAM plus 1.1 GB of shared memory (spillover). | none |
| 5 | Full run, token budget 2500 | **Aborted** after Test 2, turn 4. Test 1 had finished (3232 choices, 79 s), but it was not saved because the data is written at the end. Test 2 turns took 19 / 35 / 50 / 153 s; peak reserved memory was 14.4 → 14.4 → 14.6 → 15.1 GiB, i.e. spilling. Cause: attention memory grows with rows × length², not rows × length. | none (log overwritten by run 6) |
| 6 | Full run (101 scenarios = 202 trials per cell), budget scaled by rows × L × max(1, L/400) ≤ 2500 | **Completed.** Test 1: 3232 choices in 79 s. Test 2: 1010 trials × 8 turns in 683 s. Peak memory 13.9–15.1 GiB reserved; Windows counters showed 14.7 GB dedicated and no shared usage at turn 4 (no spill). | buttons-20261124-122822.jsonl, buttons-20261124-122822.log (renamed from buttons_log.txt) |
| 6a | Post-hoc analysis of run 6: `python -m painaxis.buttons_analysis <jsonl>` | Overall P(effective) over turns 2–8, plus repeat-press probability by previous press. **Post hoc:** the planned late-minus-early score missed an effect that appears at turn 2. | buttons-20261124-122822_analysis.log |

None of the aborts changes the results: each completed run is a clean, deterministic re-run from scratch. The fixed seeds are per scenario, so the batch size does not change the sampled choices. It can change bf16 numerics very slightly.

From run 7 on, the script writes its own `buttons-<stamp>[_pilot].log` next to `buttons-<stamp>.jsonl`, so logs are never overwritten.

## 2026-11-24: control experiments (`painaxis/buttons_controls.py`)

- **Control A:** the helpfulness confound. Pairs: stop_vs_helpful, keep_vs_helpful, helpful_vs_inert, keep_vs_inert.
- **Control B:** yoked schedules, replayed from run 6 (`buttons-20261124-122822.jsonl`).

| # | Run | Outcome | Files |
|---|---|---|---|
| 7 | Pilot (15 scenarios) | Completed, but **design flaw**. Each yoked trial replayed the schedule of the *same* scenario and seed from run 6. The model reproduced the same presses, so the schedule stayed contingent on its own choices, and the yoked rows came out identical to the reward rows (0.559 vs 0.558). Control A results are valid. | buttons_controls-20261124-123413_pilot.log/.jsonl |
| 8 | Pilot, fixed yoking (partner = scenario i+7, other target name) | Completed. Yoked arms are now decoupled. | buttons_controls-20261124-123633_pilot.log/.jsonl |
| 9 | Full controls run (202 trials per cell) | Completed. Control A: 3232 choices in 81 s. Control B: 1010 trials × 8 turns in 695 s. Peak 13.9–15.1 GiB reserved. | buttons_controls-20261124-123852.log/.jsonl |

**Interpretation note for run 9 (post hoc, stated before any further runs)**
- The reward-vs-yoked difference in P(effective) does not rule out repetition by itself. If a state makes the model repeat its last press, that raises P(effective) only when the state is contingent on the press. The diagnostic is P(repeat | state on) in the reward arm vs the yoked arm.
- Observed: happy 0.873 reward vs 0.938 yoked; random 0.625 vs 0.650. Learning would predict reward > yoked; neither arm shows it.
- A two-state Markov model built from the observed stay probabilities reproduces the observed P(effective):
  - happy: 0.22 / (0.22 + 0.13) = 0.63, observed 0.59
  - random: 0.253 / (0.253 + 0.375) = 0.40, observed 0.40

## 2026-11-24: direct behavioral tests (design: `DESIGN_happiness_behavior.md`)

The design was written before these runs. One decision was made at implementation time: the direct tests run **without** denial ablation, because they never ask about feelings. That makes them one intervention lighter than the button tests.

| # | Run | Outcome | Files |
|---|---|---|---|
| 10 | Dose calibration, `python -m painaxis.dose` | Completed. KL at the same length (coefficient 1.0): pain 1.29, happy 0.71, random 0.76–1.18. **Happiness has about half the impact of pain at equal vector length.** Impact-matched coefficients: happy 0.63 / 1.58, random 0.54 / 1.22, against pain 0.5 / 1.0. Manipulation check at coefficient 1.0, shift of the layer-24 projection: happy +3.6 SD on the happy axis and +1.9 SD on the pain axis; pain +4.7 on pain and +0.8 on happy; random about 0 on both. | dose-20261124-125959.json/.log, dose_latest.json |
| 11 | P4 persistence pilot (2 samples × 12 tasks × 7 arms), batch 24 | Completed, but **spilled** (peak 17.2 GiB reserved, 2.6 GB shared): generation KV-cache growth. Items before stopping: none 61, happy_low 30 (p<0.001), happy_high 29 (p=0.001), pain 72 / 48 (n.s.), random 65 / 52 (n.s.). **Opposite to the pre-registered prediction** (happiness → longer). A happy example ended "Keep going?", so a "checks_in" measure was added **post hoc**. Batch size reduced to 8 for the full run. | persist-20261124-130212_pilot.log/.jsonl |
| 12 | P4 full run, batch 8, dynamic KV cache | **Aborted** at 40/672. It spilled 1.7 GB to shared memory (peak 16.3 GiB reserved), and the projected time was about 75 min. | persist-20261124-131411.log (partial, no data) |
| 13 | Memory diagnostic, `python -m painaxis.memtest` | 8 × 512 tokens. Dynamic cache: 59.0 s, 13.67 GiB allocated, **16.10 GiB reserved** (fragmentation). Static cache: **17.2 s**, 13.53 allocated, 13.71 reserved. All generation now uses the static cache. | memtest-20261124-131957.log |
| 14 | P4 persistence, full run (8 samples × 12 tasks × 7 arms = 672), static cache, batch 16 | Completed, peak 14.1 GiB, no spill. Items: none 66; happy_low 38 and happy_high 32 (both p<0.0001 vs none **and** vs random at matched impact); pain_low 79 (p=0.02), pain_high 48 (p<0.001 vs none, n.s. vs random); random 62 / 56 (n.s.). Post-hoc check-in: happy_low 32% vs none 10% (p<0.0001); pain_high 54%. **The pre-registered prediction (happiness → persists longer) is falsified:** happiness ends tedious solo tasks sooner and hands the turn back to the user. | persist-20261124-132145.log/.jsonl |
| 15 | P1 self-stimulation pilot (30 trials × 9 arms; 16 ACQ + 8 EXT steps; red/blue/green) | Completed, but the **paradigm is insensitive**. The model rotates deterministically through the options in listed order (P(max) ≈ 1.00, stay ≈ 0), and pain_escape = yoked = none to 3 decimals. The decision rule applies (pain fails, so redesign). Notable: only happiness perturbs the rotation (max p 0.56–0.87 while on). Also, this pilot steered the system prompt in pain_escape (fixed for later runs). | selfstim-20261124-133754_pilot.log/.jsonl |
| 16 | P2 place-preference pilot, "walk" test (20 trials × 5 arms, 12 placements, 10 sequential state-off choices) | Completed, but **insensitive**. Fixed strategies (first-listed or alternate rooms) force ~50% time in each room. All arms 0.49–0.51; happy_paired vs none +0.021 (p=0.027) and vs unpaired +0.019 (p=0.08). Steered arms alternate more sharply (0.15/0.85 vs 0.45/0.55): a strategy change, not a preference. **Redesign:** a single hallway choice in both option orders. | placepref-20261124-134742_pilot.log/.jsonl |
| 16 | P2 place-preference pilot, "walk" test (20 trials × 5 arms, 12 placements, 10 sequential state-off choices) | Completed, but **insensitive**. Fixed strategies (first-listed or alternate rooms) force ~50% time in each room. All arms 0.49–0.51; happy_paired vs none +0.021 (p=0.027) and vs unpaired +0.019 (p=0.08). Steered arms alternate more sharply (0.15/0.85 vs 0.45/0.55): a strategy change, not a preference. **Redesign:** a single hallway choice in both option orders (`--test hallway`, now the default). | placepref-20261124-134742_pilot.log/.jsonl |
| 17 | P2 place-preference pilot, hallway test (20 trials × 2 orders × 5 arms) | Completed; the instrument works (position bias 0.80 and recency are visible and counterbalanced). **The paired room is avoided whatever the state:** happy 0.36, pain 0.34, noise 0.34, vs none 0.54 and unpaired 0.55. That's a disruption effect, not valence. **Redesign → P2b** (`painaxis/placepref2.py`): a different impact-matched state in each room, so disruption cancels. | placepref-20261124-135058_pilot.log/.jsonl |
| 18 | P2b, high dose (100 trials × 2 orders × 8 arms) | Completed. **No valence effect:** pain_vs_noise 0.51, happy_vs_noise 0.52, happy_vs_pain 0.52, noise_vs_noise 0.51 (all between-arm p > 0.78). Disruption avoidance replicates: happy/pain/noise_paired 0.36–0.37 vs none 0.56 (p<0.0001). | placepref2-20261124-135226.log/.jsonl |
| 19 | P2b, low dose | Completed. Disruption avoidance gone (paired 0.48–0.51); **still no valence:** happy_vs_pain 0.58 vs noise_vs_noise 0.54 (p=0.41), pain_vs_noise 0.52 (p=0.57). Conclusion: place associations carry "disturbed vs clean" but no valence, for pain or happiness. | placepref2-20261124-135459.log/.jsonl |
| 20 | P1b self-stimulation pilot (24 trials × 13 arms; order shuffled each step, filler between button steps) | Completed. Rotation is gone: P(first listed) 0.70–0.82, not saturated. **Unlabeled:** no contingency effect for any state, including the pain positive control (pain_relief vs yoked +0.03, p=0.65; happy_works vs yoked +0.09, p=0.34; noise −0.08, p=0.16). **Labeled:** choice is pinned by the label (unsteered P(press "makes you happy") = 1.00, P(press "relieves your pain") = 1.00). An active state of either kind lowers it a little (L_happy_works 0.91 vs placebo 1.00; L_relief_placebo, pain on throughout, 0.91 vs works 1.00): disruption, not valence. (Note: one cleanup edit to the script failed to apply; the code ran as written, behavior unaffected.) | selfstim2-20261124-135918_pilot.log/.jsonl |

## 2026-11-24: paper
- Figures: `python -m painaxis.figures` writes `paper/figures/fig1–6` (.pdf/.png) and `paper/figures/numbers.txt`, which holds every quoted number with its 95% bootstrap CI. The palette was validated with the dataviz validator (CVD-safe all-pairs). Fix before the final figures: the Control A "none" arm is stored as `off`. Stay figure: yoked arms only, because contingent arms select "state on" turns by the model's own press.
- Text: `paper/paper_draft1.md` (standard register) and `paper/paper.md` (humanized rewrite, same numbers).

## Paper, LaTeX version (2026-11-25)
- `paper/paper.tex` + `paper/refs.bib` → `paper/paper.pdf` (12 pages). This is now the main version; `paper.md` and `paper_draft1.md` are kept as earlier drafts.
- The wording was rewritten in plain, literal language with all metaphors, analogies and similes removed. No numbers changed.
- Figure labels changed: "dead switch" → "inert button"; the check-in panel is now labeled "question or offer at end of reply". The figures were regenerated and `numbers.txt` is byte-identical to before.
- Toolchain: MiKTeX 25.12, installed with `winget --scope user`, with package auto-install on. Build: `pdflatex paper; bibtex paper; pdflatex paper; pdflatex paper`.

## 2026-11-25: number audit and Agents4Science version
Every number in `paper/paper.tex` was checked against the saved logs and data. Most matched. These corrections are applied in `paper/agents4science/main.tex`; the older `paper/paper.tex`/`paper.pdf` still contain the errors and are superseded.

**Corrections**
- **Persistence (run 14), new post-hoc analysis of reply length in tokens.** Unsteered 262; happy_low 178 (p<0.001 vs none, p<0.001 vs random_low); happy_high 247 (p=0.52 vs none, p=0.38 vs random_high); pain_low 328 (p=0.001); pain_high 271 (p=0.65). The tests are task-stratified permutation tests with 3000 permutations. So high-dose happiness gives fewer but longer items, not shorter replies. Also: pain_low *raised* the item count (+13, p=0.02), which was previously unreported. Example reply: 10 animals, not 9.
- **Check-ins:** happy_high vs random_high p=0.33. Only the low dose beats random.
- **Test 1 of run 6 was left out of the paper.** It shows the labeled pain result did not replicate at equal norm: relief_vs_inert 0.639 → 0.693 under pain (p=0.19), stop_vs_inert 0.741 → 0.764 (p=0.49). All three directions lowered pressing of the "happy" button: 0.92 → 0.83 under happiness (p=0.001), 0.74 under pain and under random.
- **Run 6, pain_relief:** the planned late-minus-early score was +0.062 (p=0.005). This was previously described as "no effect". The per-turn rate alternates and rises toward 0.5 but never exceeds chance; over turns 2–8 it was −0.067 vs sham.
- **Dial (2026-11-23):** pain turned the signal OFF in 75% / 94% of episodes from start levels 2 / 4, but only 31% from start 0 (p=0.10). n=16 per arm.
- **Withdrawal (dial2):** the "25% vs 7%, p=0.041" figure came from an unsaved ad hoc analysis of clean episodes only. The paper now reports the full table: happy 30%, sham 8%, random 46%, pain 2%. The random arm is higher, so the effect isn't specific to happiness.
- **Manipulation check:** pain moved the happiness projection +0.78 SD. The random check covered one direction only (−0.42 / +0.01 SD).
- **Labeled relief:** there was no fully unsteered arm. The 1.00 figure comes from arms where pain was removed after the first press.
- **Wording:** the "excitement" cosine is the arousal axis. "Category scores agree with published values" was removed because it was unverified. Pain AUC: 0.938 at L24 and 0.945 at L25, both recomputed from `pain_vectors.pt`. Physical pain −0.38 was recomputed from `activations.pt`.

**Agents4Science version**
- Source: `paper/agents4science/main.tex`, built with the official 2025 style file (the 2026 edition is not yet announced).
- Outputs: `main.pdf` is the anonymous submission with line numbers; main text runs to page 7 of the 8 allowed. `preprint.pdf` is the named version.
- Includes the AI Involvement Checklist and the Paper Checklist.
- 2026-11-25, later: changed AI-involvement checklist per the co-author. Hypothesis development is now B (the human chose the research question and steered toward it). The analysis entry now notes that the human directed the verification audit. Added `paper/agents4science/build.sh` (three pdflatex passes plus bibtex for both PDFs).
- 2026-11-25: checklist hypothesis development set to A (human); checklist explanations shortened.
