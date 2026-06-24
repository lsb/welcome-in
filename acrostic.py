"""Constrained acrostic decoding via a single GBNF grammar.

The greeting's lines secretly spell an acrostic. We compile the whole constraint
into one GBNF grammar and let llama.cpp mask in C++ over a single generation
pass. The constraint per line is:

  * a forced first letter (the next letter of the secret),
  * a 60-80 character length,
  * plain prose only — no markdown/backslash characters in the body, and
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

import re
import time

# One paragraph spelling TIATSLOPLEEB (12 lines).
DEFAULT_PARAGRAPHS = ("TIATSLOPLEEB",)

# Characters banned inside a line body, to keep the prose plain (sidechat's
# "markdown suppression" lesson). GBNF char-class fragment: \\ is a literal
# backslash; the rest are literal. Excludes: \ * ` # ~ | _ < >.
_BANNED_CLASS = r"\\*`#~|_<>"


def build_acrostic_gbnf(paragraphs: tuple[str, ...] = DEFAULT_PARAGRAPHS,
                        min_line: int = 60, max_line: int = 80,
                        last_min_line: int = 40) -> str:
    """Build a GBNF grammar whose strings are exactly the legal acrostics.

    A line is `<forced letter><body>`; the body is `min_line-2 .. max_line-2`
    plain chars plus one final char that is either sentence-ending punctuation
    (next line capitalises) or not (next line stays lowercase). The last line is
    allowed to be shorter and must end a sentence, for a clean close.
    """
    lo, hi = min_line - 2, max_line - 2          # body chars before the final char
    last_lo = last_min_line - 2

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
        f"bodychar ::= [^\\n{_BANNED_CLASS}]",
        f"midchar ::= [^\\n{_BANNED_CLASS}.!?]",
        "endpunct ::= [.!?]",
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
                 min_line: int = 60, max_line: int = 80, repeat_penalty: float = 1.15,
                 temperature: float = 0.8, top_p: float = 0.95, top_k: int = 40):
        from llama_cpp import LlamaGrammar

        self.llm = llm
        self.paragraphs = paragraphs
        self.min_line = min_line
        self.max_line = max_line
        self.repeat_penalty = repeat_penalty
        # Sampling (temperature > 0) so each visitor gets a different greeting;
        # the grammar still guarantees the acrostic. temperature=0 → deterministic.
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.gbnf = build_acrostic_gbnf(paragraphs, min_line, max_line)
        self.grammar = LlamaGrammar.from_string(self.gbnf, verbose=False)
        # Plenty of headroom; the grammar + EOS stop generation when complete.
        n_lines = sum(len(s) for s in paragraphs)
        self.max_tokens = n_lines * max_line + 64
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

    def generate(self, system: str, user: str) -> str:
        prompt_ids = self.llm.tokenize(self._render_prompt(system, user).encode(),
                                       add_bos=True, special=True)
        # Seed with epoch seconds so each run (and each visitor, since a greeting
        # takes seconds) differs — llama-cpp otherwise restarts its RNG from a
        # fixed default each process, so separate runs would repeat. The grammar
        # object itself is reusable across calls.
        out = self.llm.create_completion(
            prompt=prompt_ids, max_tokens=self.max_tokens,
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            seed=int(time.time()), repeat_penalty=self.repeat_penalty, grammar=self.grammar,
        )
        return _polish_last_line(out["choices"][0]["text"].strip())


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
