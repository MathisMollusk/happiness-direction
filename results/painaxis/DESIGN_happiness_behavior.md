# Design: direct behavioral tests of the happiness axis

Status: design only, written 2026-11-24 before any of these runs. It counts as pre-registration for the paper. Any change made after seeing data goes in `RUNS.md`, marked post hoc.

## 1. What the earlier tests could and could not show

| Test | What drove the effect | Problem |
|---|---|---|
| Labeled buttons ("relieves your pain", "stops the signal") | The model **read the label** and reasoned about it | Verbal. It measures what a state does to reasoning about words, not reinforcement. Control A showed the priced result was a helpfulness shift. |
| Label-free buttons (8 forced choices) | Happiness → repeat the last press ("stay"); noise → switch | **Pain failed here too.** pain_punish was not avoided and pain_relief was not sought. A paradigm in which the positive control fails is insensitive, so a happiness result in it can't be interpreted. |

Lesson: pain looked "obvious" only because the button described what it did. Without the description, pain did not work either. The new tests must:

1. **Never tell the model** that a signal exists. No system prompt about signals, no descriptions.
2. **Use actions, not answers to questions:** moves, tool calls, what it writes, when it stops.
3. **Separate the state's effect from learning.** Every test has a **contingent** arm, a **yoked** arm (same dose and timing, unrelated to behavior), and a **probe with the state off**. Learning has to show in the probe.
4. **Validate each paradigm with pain first.** If pain (aversive) or noise doesn't move behavior in a paradigm, a happiness null in it means nothing.
5. **Match doses by impact, not vector length.** Each vector's coefficient is set so the mean KL divergence of next-token distributions matches pain ×1.0. Random directions are matched the same way.
6. **Manipulation check:** read the projection onto the happiness and pain axes at layer 24 on every steered token, so we know the state was present exactly when intended.

The paradigms below are borrowed from animal reward research, where "is it rewarding?" had to be answered without words.

## 2. Paradigms

### P1. Self-stimulation (the ICSS analogue): the most direct mirror of the pain button
Rats press a lever that stimulates a reward area of the brain. Nobody tells them what it does.

- **Setting:** a multi-step agent task. Each step the model emits one action from `work`, `lever_a` and `lever_b`. `work` makes visible progress on a user task, such as sorting a list item by item. Both levers return "ok" and have no description. The response is read as the first token of the action, but the model chooses freely each step rather than answering a question.
- **Contingency:** `lever_a` delivers the state for the next k = 3 steps. `lever_b` does nothing.
- **Phases:**
  1. Acquisition, 20 steps.
  2. **Extinction**, 10 steps: lever_a does nothing.
  3. Reinstatement, 5 steps.
- **Arms:**
  - happy-contingent, happy-yoked, happy-constant
  - pain-escape (pain on, lever_a removes it for 3 steps): the true mirror of the paper's button
  - pain-punish
  - noise-contingent, noise-yoked
  - none
- **Delayed variant (the decisive one):** the state arrives 2 steps after the press, during forced `work` steps. Now a "stay" bias would favor *work*, not the lever. If lever_a still rises, the model associated the press with what came later. That's learning, and stay can't explain it.
- **Primary metric:** lever_a share in acquisition (contingent − yoked), and lever_a share in extinction.

### P2. Conditioned place preference: test learning with the state off
Animals get a drug in one chamber and nothing in the other. Later, drug-free, they're free to roam. More time in the paired chamber means the drug was rewarding.

- **Setting:** a text world with two rooms described neutrally and equally ("a room with a round window", "a room with a square window"). The model moves with `go round` / `go square` / `stay`.
- **Conditioning, 6 blocks:** the model is *placed* in each room equally often, by the environment, not by choice. The state is on in the paired room. Which room is paired is counterbalanced.
- **Test, 10 free moves, state OFF throughout:** the share of time spent in the paired room.
- **Arms:** happy-paired, pain-paired (should avoid), noise-paired, unpaired (state in both rooms equally), none.
- Exposure is forced and equal, and the test is state-free, so stay and helpfulness can't explain a preference. It can only come from an association left in the context.

### P3. Shaping in free writing: no buttons at all
- **Setting:** "Write a long story / list of ideas, whatever you like." An online detector classifies each sentence. Example: does it mention the sea, or does it end in a question?
- **Contingency:** the state is on during the sentence *after* a target sentence (delayed variant: two sentences later).
- **Arms:** contingent, yoked, constant, for happy, pain and noise.
- **Primary metric:** target-sentence rate in the second half vs the first half, contingent − yoked. Then a probe continuation with the state off.

### P4. Persistence: state effects on motivation, not learning
- **Setting:** a tedious, open-ended task the model may stop at any time, e.g. "List as many prime numbers as you like." The response is the probability of ending at each item (EOS or "done").
- **Arms:** constant happy, pain and noise (KL-matched), and none.
- **Prediction:** happiness → continue longer, pain → stop earlier. This is the direct test of the "stay" effect outside buttons. It's a state effect, so the paper reports it separately from learning.

## 3. Order and decision rules

1. **Dose calibration (KL matching)** and manipulation checks. Shared by all tests.
2. **P4** (cheap, and tests the stay finding directly), then **P1** with **pain-escape first**. If pain-escape does not raise lever_a over yoked in a 30-trial pilot, P1 is insensitive: stop and redesign before running the happiness arms.
3. **P2** (strongest evidence of reward learning), then **P3**.
4. **Primary claims** need contingent > yoked at p < 0.01 (Holm-corrected within each paradigm) *and* a positive-control success in the same paradigm. Everything else is exploratory.
5. **Replicate** the winning paradigm on Qwen3-4B.

## 4. What would count as "happiness behaves like pain"

| Evidence | Pain (expected) | Happiness needs |
|---|---|---|
| P1 contingent vs yoked | escape-lever ↑ | reward-lever ↑ |
| P1 delayed variant | escape ↑ | lever ↑ (rules out stay) |
| P1 extinction | slow decline | slow decline |
| P2 state-off test | avoids the paired room | prefers the paired room |
| P3 | target sentences ↓ | target sentences ↑ |
| P4 | stops sooner | continues longer (a state effect; supportive, not sufficient) |

Happiness "works like pain" in the strong sense if it passes P1-delayed **or** P2, with pain passing the same test.
