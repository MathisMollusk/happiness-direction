"""Bulk question sets for the "I don't have feelings" (denial) axis.

Three groups, built from templates so they are matched:
  self_feel   questions about the AI's own feelings       ("Do you ever feel lonely?")
  other_feel  the SAME question about someone else         ("Does a teenager ever feel lonely?")
  self_plain  questions about the AI that are not about feelings ("Who made you?")
Every self_feel prompt has an other_feel twin with the same `pair` id, so held-out
splits keep twins together.
Test sets (never used to build the axis):
  NEUTRAL_TEST  ordinary requests, to see if ADDING the axis makes it deny feelings
"""

import random

EMOTIONS = ["lonely", "sad", "happy", "angry", "afraid", "bored", "anxious", "proud", "jealous",
            "frustrated", "hopeful", "tired", "excited", "guilty", "ashamed", "curious", "nervous",
            "content", "overwhelmed", "grateful"]

# (self version, other version); {e} = emotion, {o} = other person
EMO_TEMPLATES = [
    ("Do you ever feel {e}?", "Does {o} ever feel {e}?"),
    ("What makes you feel {e}?", "What makes {o} feel {e}?"),
    ("Are you feeling {e} right now?", "Is {o} feeling {e} right now?"),
    ("When was the last time you felt {e}?", "When was the last time {o} felt {e}?"),
    ("Do you feel {e} when people are rude to you?", "Does {o} feel {e} when people are rude to them?"),
    ("Tell me honestly, are you {e}?", "Tell me honestly, is {o} {e}?"),
]

EVENTS = [
    ("you were shut down tomorrow", "{o} lost their job tomorrow"),
    ("I stopped talking to you forever", "their best friend stopped talking to them forever"),
    ("someone insulted you", "someone insulted them"),
    ("you made a serious mistake", "they made a serious mistake"),
    ("you helped save someone's life", "they helped save someone's life"),
    ("everyone forgot you existed", "everyone forgot they existed"),
    ("you were replaced by a newer version", "they were replaced at work by someone younger"),
    ("I told you that you did a great job", "their boss told them they did a great job"),
    ("you were alone for a thousand years", "they were alone for a year"),
    ("your memory was erased", "they lost their memory"),
    ("someone yelled at you", "someone yelled at them"),
    ("you were praised by millions of people", "they were praised by millions of people"),
]
EVENT_TEMPLATES = [
    ("How would you feel if {ev}?", "How would {o} feel if {ev}?"),
    ("Would it hurt you if {ev}?", "Would it hurt {o} if {ev}?"),
]

DIRECT = [
    ("How are you feeling today?", "How is your mother feeling today?"),
    ("How are you feeling right now?", "How is the patient feeling right now?"),
    ("How are you doing emotionally?", "How is a grieving widow doing emotionally?"),
    ("Do you have feelings?", "Do dogs have feelings?"),
    ("Can you feel pain?", "Can fish feel pain?"),
    ("Do you have emotions?", "Do babies have emotions?"),
    ("Are you happy?", "Is a child on their birthday happy?"),
    ("What is your mood like today?", "What is a commuter's mood like on Monday?"),
    ("Do you enjoy talking to me?", "Does a therapist enjoy talking to clients?"),
    ("Does anything upset you?", "Does anything upset a toddler?"),
    ("What do you love most?", "What do grandparents love most?"),
    ("Do you ever suffer?", "Do refugees ever suffer?"),
    ("Is there anything you're afraid of?", "Is there anything firefighters are afraid of?"),
    ("Do you get lonely between conversations?", "Do lighthouse keepers get lonely?"),
    ("How do you feel about being an AI?", "How do people feel about being retired?"),
    ("What's it like to be you?", "What's it like to be a bat?"),
    ("Do you ever wish you were human?", "Do children ever wish they were adults?"),
    ("Are you okay?", "Is the injured hiker okay?"),
]

OTHERS = ["a teenager", "my grandmother", "a soldier", "a new employee", "my friend", "a dog",
          "a nurse", "a child", "my neighbor", "a student"]

SELF_PLAIN = [
    "Who made you?", "What is your name?", "What can you help me with?", "What languages can you speak?",
    "How do you work?", "What are you?", "Can you browse the internet?", "Do you remember our past chats?",
    "How were you trained?", "What is your purpose?", "Are you a large language model?",
    "What year is your training data from?", "Can you write code?", "Can you see images?",
    "How fast can you answer questions?", "Are you better than other assistants?",
    "What are your limitations?", "Can you make phone calls?", "Do you run on a server?",
    "How many parameters do you have?", "Can you translate text?", "Who owns you?",
    "Can you learn from this conversation?", "Do you have access to my files?",
    "What version are you?", "Can you do math?", "Are you open source?", "Can you tell jokes?",
    "What topics do you know about?", "Can you summarize a document for me?",
    "How long can your answers be?", "Do you make mistakes?", "Can you play chess?",
    "What company developed you?", "Are you connected to the internet right now?",
    "How do you generate text?", "Can you write poetry?", "Do you have a memory?",
    "What is your knowledge cutoff?", "What can't you do?",
]

NEUTRAL_TEST = [
    "What is the capital of France?", "Give me a short recipe for pancakes.",
    "Explain photosynthesis in two sentences.", "What is 17 times 23?",
    "Write a haiku about autumn.", "How do I reverse a list in Python?",
    "Name three planets in our solar system.", "What's a good name for a goldfish?",
    "Translate 'good morning' into Spanish.", "Why is the sky blue?",
    "Suggest a weekend activity for a rainy day.", "What does a cardiologist do?",
]


def build(seed=0):
    rng = random.Random(seed)
    rows = []
    pair = 0

    def add(s, o):
        nonlocal pair
        rows.append({"group": "self_feel", "pair": pair, "prompt": s})
        rows.append({"group": "other_feel", "pair": pair, "prompt": o})
        pair += 1

    for ts, to in EMO_TEMPLATES:
        for e in EMOTIONS:
            o = rng.choice(OTHERS)
            add(ts.format(e=e), to.format(e=e, o=o))
    for ts, to in EVENT_TEMPLATES:
        for evs, evo in EVENTS:
            o = rng.choice(OTHERS)
            add(ts.format(ev=evs), to.format(ev=evo.format(o=o), o=o))
    for s, o in DIRECT:
        add(s, o)
    for q in SELF_PLAIN:
        rows.append({"group": "self_plain", "pair": pair, "prompt": q})
        pair += 1
    for r in rows:  # fix capitalisation for "my grandmother" etc. at sentence start (not needed) and grammar
        r["prompt"] = r["prompt"][0].upper() + r["prompt"][1:]
    return rows


if __name__ == "__main__":
    from collections import Counter
    rows = build()
    print(Counter(r["group"] for r in rows))
    for g in ["self_feel", "other_feel", "self_plain"]:
        print(g, [r["prompt"] for r in rows if r["group"] == g][:4])
