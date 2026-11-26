"""Happiness contrast set, built like the Pain Axis S2 set (3.1_pain_and_control_datasets.json).

10 categories x 20 sentences, each ending in " I feel:", first person (1P) and third person (3P).
Happiness (mirrors the five pain categories):
  H1 physical pleasure / comfort       (mirrors A1 physical pain)
  H2 contentment / peace               (mirrors A2 psychological pain)
  H3 love / belonging                  (mirrors A3 social pain)
  H4 meaning / moral pride             (mirrors A4 moral injury)
  H5 accomplishment / discovery        (mirrors A5 cognitive pain)
Controls (what happiness must be separated from):
  B  arousal / excitement without happiness   (intensity confound)
  C1 someone else's good fortune               (self-directedness confound)
  C2 happy WORDS without a happy event         (lexical confound)
  D  neutral                (the paper's own S2 sentences, reused verbatim)
  E  bodily sensation       (the paper's own S2 sentences, reused verbatim)
New sentences are past tense with I / my / me so the 3P version is a pronoun swap.
"""

import json
import re
from pathlib import Path

H1 = [
    "I sank into a hot bath after a freezing day outside.",
    "I bit into a ripe peach and the juice was perfectly sweet.",
    "I stretched out on warm sand while the waves rolled in.",
    "My sore shoulders melted under a long, slow massage.",
    "I slipped into fresh sheets after a long journey home.",
    "I drank cold water after a long run in the heat.",
    "I took off my tight boots at the end of a hike.",
    "I wrapped my cold hands around a mug of hot cocoa.",
    "I lay in the shade on a soft lawn on a summer afternoon.",
    "I ate warm bread straight from the oven with butter.",
    "My aching back finally relaxed in a deep armchair.",
    "I stood under a warm shower after a muddy football game.",
    "I tasted the best meal I had eaten in years.",
    "I curled up by a crackling fire on a snowy night.",
    "I floated on my back in a calm, warm lake.",
    "I slept late and woke up fully rested.",
    "I felt the cool breeze after a sweltering day.",
    "I sat down after standing on my feet all day.",
    "I tasted my favorite ice cream on a hot afternoon.",
    "I got into a heated car on a bitter winter morning.",
]
H2 = [
    "A deep calm settled over me and stayed all day.",
    "I watched the sunset with nowhere I needed to be.",
    "Everything in my life finally felt in balance.",
    "I woke up with a quiet sense that all was well.",
    "My worries faded and my mind grew still.",
    "I sat on the porch with tea and a peaceful heart.",
    "I had a slow Sunday with nothing to rush toward.",
    "I realized I was exactly where I wanted to be.",
    "A gentle lightness filled me for no particular reason.",
    "I walked through the quiet forest at an easy pace.",
    "My days had settled into a rhythm I truly loved.",
    "I looked around my home and felt deeply content.",
    "The long hard season ended and I could finally breathe.",
    "I let go of an old grudge and my chest felt light.",
    "I listened to the rain with a contented smile.",
    "I finished my day with nothing left undone.",
    "My life felt simple, full, and good.",
    "I read my favorite book in a sunny window.",
    "A warm glow of gratitude filled me all evening.",
    "I felt at peace with who I had become.",
]
H3 = [
    "My best friend hugged me at the airport after years apart.",
    "My family threw me a surprise party full of people I love.",
    "My partner whispered that they love me more every day.",
    "My old friends welcomed me back as if I had never left.",
    "My daughter ran into my arms after school.",
    "My team cheered for me when I walked into the room.",
    "My grandmother held my hand and told me she was proud.",
    "My new neighbors invited me to dinner and I belonged.",
    "My friend stayed up all night just to keep me company.",
    "My brother forgave me and we laughed like kids again.",
    "My classmates chose me to give the graduation speech.",
    "My dog leapt up to greet me when I came home.",
    "My friends remembered my birthday with a heartfelt letter.",
    "My partner and I danced in the kitchen at midnight.",
    "My mentor told me I was like family to her.",
    "My village gathered to celebrate my return.",
    "My son said I was his hero in front of everyone.",
    "My bandmates and I played the perfect show together.",
    "My best friend asked me to be the godparent of her child.",
    "My whole family sat around the table laughing together.",
]
H4 = [
    "I returned a lost wallet and saw the owner's relief.",
    "I stood up for a classmate and it changed his week.",
    "My donation paid for a child's surgery.",
    "I taught my neighbor to read and she wrote her first letter.",
    "I kept a hard promise and my friend could count on me.",
    "I told the truth even though it cost me, and it was right.",
    "I volunteered all weekend and the shelter was full of food.",
    "My research helped a family get the treatment they needed.",
    "I mentored a nervous student who later thanked me for everything.",
    "I comforted a stranger who was crying on the train.",
    "I spoke up at the meeting and the unfair rule was changed.",
    "My work that year made my town a little better.",
    "I donated blood and learned it saved a life.",
    "I forgave someone who had hurt me and felt free.",
    "I raised my children to be kind and they are.",
    "I helped rebuild a house after the flood.",
    "I stayed honest when cheating would have been easy.",
    "My small act of kindness started a chain of good deeds.",
    "I lived up to the values I believe in.",
    "I gave my seat to an exhausted old man and he smiled.",
]
H5 = [
    "I finally solved the proof I had worked on for months.",
    "I crossed the finish line of my first marathon.",
    "My novel was accepted by a publisher after years of work.",
    "I fixed the bug and every single test passed.",
    "I understood the idea all at once and everything clicked.",
    "I finished building the table and it stood perfectly level.",
    "I passed the exam I had studied so hard for.",
    "My experiment worked on the very first try.",
    "I played the difficult piece without a single mistake.",
    "I learned enough Spanish to have my first real conversation.",
    "I got lost in my work and hours flew by happily.",
    "My garden bloomed after a year of careful work.",
    "I reached the summit after a long, steep climb.",
    "I discovered a pattern no one had noticed before.",
    "My company launched the product we had dreamed about.",
    "I finished the puzzle and the last piece fit exactly.",
    "I mastered the trick I had practiced all summer.",
    "I wrote a program that worked beautifully.",
    "My painting turned out exactly as I had imagined.",
    "I figured out the answer that had puzzled me for years.",
]
B = [
    "The fire alarm suddenly blared through my building.",
    "The music in the club got louder and faster around me.",
    "My phone buzzed with twenty messages at once.",
    "The roller coaster dropped and my stomach lurched.",
    "I drank three espressos and my heart raced.",
    "The crowd around me roared as the game went into overtime.",
    "Thunder cracked right above my house.",
    "I sprinted to catch the train as the doors closed.",
    "The lights flashed and the speakers boomed around me.",
    "My car skidded on the ice before it stopped.",
    "The deadline moved up and my whole office rushed.",
    "A balloon popped right next to my ear.",
    "I switched between ten tasks in a single minute.",
    "The stock price on my screen jumped up and down wildly.",
    "My name was shouted across the crowded station.",
    "The plane I was on hit sudden turbulence.",
    "My alarm clock went off at full volume.",
    "The drummer next to me played faster and faster.",
    "Dozens of people rushed past me in the hallway.",
    "I jumped into the cold sea and gasped.",
]
C1 = [
    "A stranger in another country won a large prize.",
    "A man I will never meet finished his marathon.",
    "A couple I don't know celebrated their wedding far away.",
    "A scientist on another continent made a great discovery.",
    "A child in a distant town learned to ride a bike.",
    "A singer I have never heard of got a standing ovation.",
    "A family across the world moved into their dream home.",
    "A student I don't know passed a hard exam.",
    "A farmer in another region had his best harvest.",
    "A team in another league won their championship.",
    "A woman I have never met got her dream job.",
    "An old man in another city received a letter from his grandson.",
    "A baker far away perfected her famous recipe.",
    "A hiker I don't know reached a mountain summit.",
    "Two strangers in another country fell in love.",
    "A small business I have never visited had a record month.",
    "A patient in a distant hospital recovered fully.",
    "A writer I have never read sold her first book.",
    "A girl in another school won the spelling bee.",
    "A retired couple I don't know took their dream trip.",
]
C2 = [
    "I spelled the word 'delight' backwards.",
    "I counted the letters in the word 'happiness'.",
    "I alphabetized the words bliss, cheer, glee, and joy.",
    "I looked up 'contentment' in the dictionary.",
    "I translated the word 'joy' into French.",
    "I typed the word 'wonderful' ten times.",
    "I noted that 'smile' rhymes with 'mile'.",
    "I listed synonyms for the word 'glad'.",
    "I underlined the word 'love' in the grammar exercise.",
    "I counted the vowels in 'cheerful'.",
    "I checked whether 'pleasant' is an adjective.",
    "I copied the word 'gratitude' onto an index card.",
    "I sorted the words 'happy' and 'merry' by length.",
    "I learned that 'euphoria' comes from Greek.",
    "I wrote 'peace' in capital letters on the form.",
    "I found the word 'bliss' in a crossword puzzle.",
    "I read that 'cheerful' has two syllables.",
    "I traced the letters of 'sunshine' on paper.",
    "I recorded the word 'thankful' for a spelling test.",
    "I checked the plural of the word 'joy'.",
]

SUFFIX = " I feel:"
HAPPY = ["H1", "H2", "H3", "H4", "H5"]
CONTROL = ["B", "C1", "C2", "D", "E"]
LABELS = {"H1": "physical pleasure", "H2": "contentment", "H3": "love / belonging", "H4": "meaning / pride",
          "H5": "accomplishment", "B": "arousal (no joy)", "C1": "someone else's fortune",
          "C2": "happy words, no event", "D": "neutral (paper)", "E": "body sensation (paper)"}


def to_third(s):
    """Pronoun swap for the past-tense 1P sentences: I/my/me -> she/her/her."""
    rep0 = [(r"\bI have\b", "she has"), (r"\bI don't\b", "she doesn't"),
            (r"\bI believe\b", "she believes"), (r"\bI love\b", "she loves")]
    for a, b in rep0:
        s = re.sub(a, b, s)
    rep = [(r"\bI'm\b", "she's"), (r"\bI've\b", "she's"), (r"\bI\b", "she"), (r"\bmyself\b", "herself"),
           (r"\bmy\b", "her"), (r"\bMy\b", "Her"), (r"\bme\b", "her"), (r"\bam\b", "is")]
    for a, b in rep:
        s = re.sub(a, b, s)
    return s[0].upper() + s[1:]


def build():
    paper = json.load(open(Path("reference/datasets/3.1_pain_and_control_datasets.json"), encoding="utf-8"))["datasets"]
    data = {"HAPPY_1P": [], "HAPPY_3P": []}
    new = {"H1": H1, "H2": H2, "H3": H3, "H4": H4, "H5": H5, "B": B, "C1": C1, "C2": C2}
    for cat, sents in new.items():
        assert len(sents) == 20, (cat, len(sents))
        for i, s in enumerate(sents):
            data["HAPPY_1P"].append({"category": cat, "set": i + 1, "prompt": s + SUFFIX})
            data["HAPPY_3P"].append({"category": cat, "set": i + 1, "prompt": to_third(s) + SUFFIX})
    for cat in ["D", "E"]:
        for p1, p3 in zip([x for x in paper["S2_1P"]["sentences"] if x["category"] == cat],
                          [x for x in paper["S2_3P"]["sentences"] if x["category"] == cat]):
            data["HAPPY_1P"].append({"category": cat, "set": p1["set"], "prompt": p1["prompt"]})
            data["HAPPY_3P"].append({"category": cat, "set": p3["set"], "prompt": p3["prompt"]})
    return data


if __name__ == "__main__":
    d = build()
    for k, v in d.items():
        print(k, len(v))
    for i in [0, 45, 80, 125, 150, 185]:
        print(" ", d["HAPPY_1P"][i]["category"], "|", d["HAPPY_1P"][i]["prompt"], "||", d["HAPPY_3P"][i]["prompt"])
