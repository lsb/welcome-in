"""Stage 3 — the visitor question card.

After the host's brief hello (greeter.py, Part 1), the gallery does not lecture the
visitor — it asks. Part 2 is a short card of open questions about generative AI, on
one randomly chosen theme (TOPICS), so visitors argue and wonder for themselves
instead of watching passively. It is generated under the acrostic mask (acrostic.py)
as a fresh conversation with its own system prompt; the acrostic's length suits a
handful of questions, and the grammar now permits '?'.
"""

from __future__ import annotations

import random

# One of these seventeen themes is chosen at random per visitor; the model writes a
# few questions that open it up, rather than answering it.
TOPICS = (
    "What is generative AI content like ChatGPT as an aesthetic condition?",
    "What is generative AI content like ChatGPT as a social condition?",
    "What is generative AI content like ChatGPT as an economic condition?",
    "What is generative AI content like ChatGPT as a psychological condition?",
    "What is an embodiment of generative AI content like ChatGPT?",
    "What is the weaponization of generative AI content like ChatGPT?",
    "What is a surrender to generative AI content like ChatGPT?",
    "What is the role of surveillance and algorithmic visibility in generative AI content like ChatGPT?",
    "What is the role of invisible digital labor in generative AI content like ChatGPT?",
    "What is the role of authorship, attribution, and remix culture in generative AI content like ChatGPT?",
    "What is the role of realism, hallucination, and machine perception in generative AI content like ChatGPT?",
    "What is the role of trust, credibility, and post-truth aesthetics in generative AI content like ChatGPT?",
    "What is the role of attention, engagement, and optimization in generative AI content like ChatGPT?",
    "What is the role of meme propagation and information decay in generative AI content like ChatGPT?",
    "What is the role of synthetic intimacy and emotional attachment to generated systems in generative AI content like ChatGPT?",
    "What is the role of glitches, compression artifacts, and degraded media in generative AI content like ChatGPT?",
    "What is the role of automation, bureaucracy, and interface rituals in generative AI content like ChatGPT?",
)

# A fresh chat. The point is engagement, not answers — visitors should leave with
# questions to argue about. The example questions carry the register (short, direct,
# second-person, everyday); they are on a different theme so they bleed obviously.
QUESTIONS_SYSTEM = (
    "You write the question card on the wall beside the works in an art gallery full of "
    "artists working with generative AI content like ChatGPT. For the theme you are "
    "given, write a handful of short, open questions that pull visitors in and make them "
    "argue, wonder, and take sides — questions to turn to the stranger beside you and "
    "ask, not answers to nod along to. Keep each one plain, direct, and rooted in "
    "everyday experience. Only questions; no preamble, no answers, no headings.\n\n"
    "For the tone (a different theme):\n"
    "When a machine makes the picture in a second, do you lean in closer or walk away? "
    "Who do you trust less, the image or the person who typed the words behind it? "
    "If everyone can make this, is any of it still worth stopping for? /no_think"
)


def pick_topic(rng: random.Random | None = None) -> str:
    """Choose one of the seventeen themes at random (one per visitor)."""
    return (rng or random).choice(TOPICS)


def build_questions_prompt(topic: str) -> str:
    """The per-card user turn."""
    return f"Theme: {topic}\nWrite the question card for visitors."
