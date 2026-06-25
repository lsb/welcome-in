"""Constrained acrostic decoding via a single GBNF grammar.

The greeting's lines secretly spell an acrostic. We compile the whole constraint
into one GBNF grammar and let llama.cpp mask in C++ over a single generation
pass. The constraint per line is:

  * a forced first letter (the next letter of the secret),
  * a per-line length (default 10-100 chars; tunable, see acrostic_from_env),
  * a plain-English character whitelist (letters, spaces, and a little
    punctuation; sentences end on a period or a question mark) — this is also the
    project's most reliable tone control, since it makes whole classes of
    off-register output impossible on a tiny model: no digits (hence no invented
    prices or dates), no emojis, no stray non-English scripts, no markdown, and no
    '!' (so it cannot exclaim or gush). '?' is allowed: the acrostic now carries
    the visitor question card (questions.py), which is all questions. See _BODY_CHARS.
  * stealth casing: the forced letter is lowercase mid-sentence (so the prose
    flows and the acrostic hides), uppercase only at the start of a sentence.

This is the grammar-masking idea from github.com/lsb/sidechat (its grammar.js /
logits.js), expressed declaratively: the constraint — casing included — is a
regular language, so a grammar captures it directly. Stealth casing is encoded
in continuation-passing style — each line branches the next line's first-letter
case on whether it ended a sentence:

    segIcap ::= "<Upper>" bodyHead ( endpunct SEP segJcap | midchar SEP segJlow )
    segIlow ::= "<lower>" bodyHead ( endpunct SEP segJcap | midchar SEP segJlow )

(An earlier version did per-line decoding plus a local-crossing search for nicer
line breaks; on this model that matched greedy quality at ~3.5x the latency, and
the run is model-bound anyway, so we dropped it for this one-call grammar.)
"""

from __future__ import annotations

import os
import re
import time

# One paragraph spelling TIATSLOPLEEB (12 lines), 10-100 chars each. These three
# defaults define the acrostic's shape and are all overridable from the
# environment (see acrostic_from_env) — the shape is the strongest lever on the
# greeting's register, so being able to tune it without code edits matters: a
# short secret lets the host say its piece and stop (crisp), a long one forces it
# to keep talking past its content (padded). See greeting-tone notes in greeter.py.
DEFAULT_PARAGRAPHS = ("TIATSLOPLEEB",)
DEFAULT_MIN_LINE = 10
DEFAULT_MAX_LINE = 100

# Environment variables that override the above (read once per Greeter).
ACROSTIC_ENV = "WELCOME_ACROSTIC"     # the secret to spell; '|' splits paragraphs
MIN_LINE_ENV = "WELCOME_MIN_LINE"
MAX_LINE_ENV = "WELCOME_MAX_LINE"

# WELCOME_ACROSTIC values that switch the acrostic off entirely -> raw, masked-by-
# nothing model output (markdown, digits, hedging and all). Signalled downstream
# as an empty paragraphs tuple.
ACROSTIC_OFF = frozenset({"off", "none", "no", "raw"})


def acrostic_from_env() -> tuple[tuple[str, ...], int, int]:
    """Resolve (paragraphs, min_line, max_line) for the acrostic constraint,
    letting the environment override the module defaults:

      WELCOME_ACROSTIC   secret to spell, ASCII letters only; '|' separates
                         paragraphs, e.g. "OPEN" or "HELLO|FRIEND". Other
                         characters are dropped; empty/unset -> the TIATSLOPLEEB
                         default. Set to "off" (or none/no/raw) to drop the grammar
                         mask entirely and get the model's raw, unconstrained slop.
      WELCOME_MIN_LINE   minimum characters per line (default 10).
      WELCOME_MAX_LINE   maximum characters per line (default 100).

    Returns paragraphs == () when the acrostic is switched off.

    Examples:
      WELCOME_ACROSTIC=OPEN WELCOME_MIN_LINE=32 WELCOME_MAX_LINE=48 \\
        uv run python main.py 2.png      # terser, closer to plain speech
      WELCOME_ACROSTIC=off uv run python main.py 2.png   # raw model output
    """
    raw = os.environ.get(ACROSTIC_ENV, "")
    if raw.strip().lower() in ACROSTIC_OFF:
        paragraphs: tuple[str, ...] = ()     # () == acrostic disabled (raw output)
    else:
        paragraphs = tuple(
            cleaned
            for part in raw.split("|")
            if (cleaned := "".join(c for c in part if c.isascii() and c.isalpha()))
        ) or DEFAULT_PARAGRAPHS
    min_line = int(os.environ.get(MIN_LINE_ENV, DEFAULT_MIN_LINE))
    max_line = int(os.environ.get(MAX_LINE_ENV, DEFAULT_MAX_LINE))
    return paragraphs, min_line, max_line

# Allowed characters inside a line body — a whitelist, not a blacklist. On a tiny
# model the grammar mask is the only *reliable* tone control, so we let it enforce
# crisp, professional copy at the character level: plain English letters, spaces,
# and a small punctuation set. This structurally rules out whole failure modes a
# prompt can't reliably suppress on a 0.8B model — no digits (so no invented
# prices, percentages, or dates), no emojis, no stray non-English scripts (the
# Qwen base leaks CJK under sampling), and no markdown. Sentences end on a period
# or a question mark (see endpunct) — '?' is wanted now that the acrostic carries
# the visitor question card — but never '!', so it cannot exclaim or gush.
# (sidechat's "markdown suppression" lesson, taken to its logical end.)
_BODY_CHARS = r"a-zA-Z ,.;:'?-"     # general body char; may include a mid-line . or ?
_MID_CHARS = r"a-zA-Z ,;:'-"        # body char that does NOT end a sentence (no . or ?)


def build_acrostic_gbnf(paragraphs: tuple[str, ...] = DEFAULT_PARAGRAPHS,
                        min_line: int = DEFAULT_MIN_LINE, max_line: int = DEFAULT_MAX_LINE,
                        last_min_line: int = 40) -> str:
    """Build a GBNF grammar whose strings are exactly the legal acrostics.

    A line is `<forced letter><body>`; the body is `min_line-2 .. max_line-2`
    plain chars plus one final char that is either sentence-ending punctuation
    (next line capitalises) or not (next line stays lowercase). The last line is
    allowed to be shorter and must end a sentence, for a clean close.
    """
    # body chars before the final char. Clamped to 0 <= lo,last_lo <= hi so that
    # out-of-range env tuning (e.g. a tiny max_line) can't emit an invalid GBNF
    # repetition like bodychar{38,28}. last_lo is also capped at lo, so the closing
    # line is never forced *longer* than a regular line (it may still be shorter).
    hi = max(0, max_line - 2)
    lo = min(max(0, min_line - 2), hi)
    last_lo = min(max(0, last_min_line - 2), lo)

    # Flatten paragraphs into (forced letter, separator-after-this-line).
    plan: list[tuple[str, str | None]] = []
    for p, secret in enumerate(paragraphs):
        for i, letter in enumerate(secret):
            last = (p == len(paragraphs) - 1) and (i == len(secret) - 1)
            if last:
                sep = None
            elif i == len(secret) - 1:
                sep = r"\n\n"   # paragraph break
            else:
                sep = r"\n"     # line break within a paragraph
            plan.append((letter, sep))
    n = len(plan)

    rules = [
        f"bodychar ::= [{_BODY_CHARS}]",
        f"midchar ::= [{_MID_CHARS}]",
        "endpunct ::= [.?]",
        f"bodyHead ::= bodychar{{{lo},{hi}}}",
        "root ::= seg0cap",
    ]
    for i, (letter, sep) in enumerate(plan):
        upper, lower = letter.upper(), letter.lower()
        if i == n - 1:  # last line: shorter ok, must close a sentence, then EOS
            rules.append(f'seg{i}cap ::= "{upper}" bodychar{{{last_lo},{hi}}} endpunct')
            rules.append(f'seg{i}low ::= "{lower}" bodychar{{{last_lo},{hi}}} endpunct')
        else:
            tail = f'( endpunct "{sep}" seg{i+1}cap | midchar "{sep}" seg{i+1}low )'
            rules.append(f'seg{i}cap ::= "{upper}" bodyHead {tail}')
            rules.append(f'seg{i}low ::= "{lower}" bodyHead {tail}')
    return "\n".join(rules)


class AcrosticDecoder:
    """Generates an acrostic greeting from a chat prompt in one grammar-masked
    pass. Compile once (cheap); reuse `generate()` per visitor."""

    def __init__(self, llm, paragraphs: tuple[str, ...] = DEFAULT_PARAGRAPHS,
                 min_line: int = DEFAULT_MIN_LINE, max_line: int = DEFAULT_MAX_LINE,
                 repeat_penalty: float = 1.2,
                 temperature: float = 0.3, top_p: float = 0.95, top_k: int = 30,
                 frequency_penalty: float = 0.4, presence_penalty: float = 0.0):
        from llama_cpp import LlamaGrammar

        self.llm = llm
        self.paragraphs = paragraphs
        self.min_line = min_line
        self.max_line = max_line
        # The acrostic forces ~800 characters of prose out of a 0.8B model that
        # only has a few true things to say, so left alone it pads and sometimes
        # loops ("Enjoy it. Let's go." over and over). A modest temperature keeps
        # it from wandering/inventing; the repeat + frequency penalties keep it
        # from circling the same phrases while it fills the grammar's length.
        self.repeat_penalty = repeat_penalty
        # Sampling (temperature > 0) so each visitor gets a different greeting;
        # the grammar still guarantees the acrostic. temperature=0 → deterministic.
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        if paragraphs:
            self.gbnf = build_acrostic_gbnf(paragraphs, min_line, max_line)
            self.grammar = LlamaGrammar.from_string(self.gbnf, verbose=False)
            # Plenty of headroom; the grammar + EOS stop generation when complete.
            n_lines = sum(len(s) for s in paragraphs)
            self.max_tokens = n_lines * max_line + 64
        else:
            # Acrostic disabled (paragraphs == ()): no grammar mask at all, so the
            # tiny model's raw, unconstrained output comes straight through —
            # markdown, digits, hedging and all. EOS usually stops it; the cap is
            # just a backstop against a rambler.
            self.gbnf = None
            self.grammar = None
            self.max_tokens = 512
        self._formatter = None  # lazy chat formatter

    def _render_prompt(self, system: str, user: str) -> str:
        if self._formatter is None:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            md = self.llm.metadata
            self._formatter = Jinja2ChatFormatter(
                template=md["tokenizer.chat_template"],
                eos_token="<|im_end|>", bos_token="", add_generation_prompt=True,
            )
        return self._formatter(messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]).prompt

    def generate(self, system: str, user: str, seed: int | None = None,
                 on_delta=None) -> str:
        prompt_ids = self.llm.tokenize(self._render_prompt(system, user).encode(),
                                       add_bos=True, special=True)
        # Seed with epoch seconds so each run (and each visitor, since a greeting
        # takes seconds) differs — llama-cpp otherwise restarts its RNG from a
        # fixed default each process, so separate runs would repeat. The grammar
        # object itself is reusable across calls. Callers (e.g. the tone harness)
        # may pass an explicit seed to get distinct samples within one second.
        kw = dict(
            prompt=prompt_ids, max_tokens=self.max_tokens,
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            seed=int(time.time()) if seed is None else seed,
            repeat_penalty=self.repeat_penalty,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            grammar=self.grammar,   # None when the acrostic is switched off
        )
        if on_delta is None:
            text = self.llm.create_completion(**kw)["choices"][0]["text"]
        else:
            # Stream token-by-token so a live display can render the greeting as
            # the model writes it — on the Pi each part takes tens of seconds, so
            # the wait *is* the show. The grammar still masks every token; on_delta
            # sees raw pieces, and _polish_last_line settles the forced last-line
            # fragment once generation completes (so the screen shows live text,
            # then snaps to the same finished string the printer/return value use).
            pieces: list[str] = []
            for chunk in self.llm.create_completion(stream=True, **kw):
                piece = chunk["choices"][0]["text"]
                if piece:
                    pieces.append(piece)
                    on_delta(piece)
            text = "".join(pieces)
        text = text.strip()
        # _polish_last_line tidies the acrostic's forced final line; with no
        # grammar there's no such structure, so return the raw text untouched.
        return text if self.grammar is None else _polish_last_line(text)


def _polish_last_line(text: str) -> str:
    """The last line's terminal punctuation is forced at the char wall, so it can
    land mid-thought (`…stories waiting to be told. You feel like stepping on.`).
    If the line has an earlier sentence end, drop the trailing fragment so the
    greeting closes on a complete sentence — the forced first letter sits well
    before any such point, so the acrostic is preserved. Otherwise just tidy a
    dangling space/comma before the period."""
    lines = text.split("\n")
    last = lines[-1].rstrip()
    inner = max((last.rfind(c, 0, len(last) - 1) for c in ".!?"), default=-1)
    if inner != -1:
        lines[-1] = last[:inner + 1]
    else:
        lines[-1] = re.sub(r"[\s,;:]+([.!?])$", r"\1", last)
    return "\n".join(lines)
