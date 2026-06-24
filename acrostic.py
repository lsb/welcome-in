"""Constrained acrostic decoding — a CPU/llama-cpp port of github.com/lsb/sidechat.

Sidechat (see its ACROSTIC_DECODING_SEARCH.md) decodes text whose lines spell a
secret word. Three of its ideas are carried over here, adapted to a Raspberry-Pi
deployment running Qwen3.5 GGUF through llama-cpp-python:

  * Grammar-masked decoding (sidechat: grammar.js + logits.js). At every step we
    mask every token that would break the per-line constraint — the forced first
    letter and the min/max line length — so anything sampled is a valid acrostic
    prefix. No post-hoc validation, no gobbledygook.

  * Local-crossing search (sidechat: crossingSearch.js). Generate each line
    greedily, then make the *break point* a search variable: try ending the line
    a few tokens earlier, at a word boundary, and score only a short window
    straddling the break (the last `k` log-probs before it, the forced letter,
    and the next `j` tokens). The objective rewards breaks where the next forced
    letter becomes the *natural next word*, not breaks that merely run to the
    wall. r=0 (the greedy break) is always a candidate, so search only ever
    matches or beats greedy.

  * Stealth casing + a min-line floor (sidechat: surprisalLookahead.js). Mid-
    sentence, force the forced letter lowercase so the prose flows and the
    acrostic hides down the margin; never let a forced letter produce a stubby
    line.

The one deliberate divergence: sidechat walks a per-token NFA over its ~50k
vocab, which is cheap in JS. Qwen3.5's vocab is ~248k, so here the mask is a
numpy boolean op over per-token arrays precomputed once at load — a few hundred
KB, ~1-3 ms/step on a Pi, negligible beside the model's forward pass (whereas
the per-token scan would rival it). The grammar itself is so simple (one forced
letter, then body, per line) that the masker computes legality directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Tokens that may legally START a trimmed-off tail in the crossing search — i.e.
# a clean word/punctuation boundary, so trimming never splits a word.
_BREAK_TOKEN = re.compile(r"^[\s.,;:!?)\]\"'’”]")

# Plain-prose enforcement (sidechat's "markdown suppression" lesson, but applied
# in the mask rather than only the prompt — a 0.8B under heavy constraint will
# otherwise sprinkle markdown hard-breaks `\` and emphasis `*` `_` etc.). Any
# token containing one of these is masked out of the body.
_BANNED_CHARS = set("\\*`#~|_<>")
# A run of text ends a sentence if its last non-space char is . ! or ? (with an
# optional closing quote/bracket). Then the next line may start with a capital.
_SENT_END = re.compile(r"[.!?][\"'”’)\]]?$")


def mid_sentence(text: str) -> bool:
    """True if `text` does not end a sentence — so the next forced letter should
    be lowercase (grammatically right, and keeps the acrostic hidden). An empty
    prefix is a sentence start, so capitals are fine there."""
    s = text.rstrip()
    return len(s) > 0 and _SENT_END.search(s) is None


@dataclass
class _TokenTable:
    """Per-token arrays over the whole vocab, built once. Everything the mask
    needs is a vectorised lookup into these."""

    txt: list[str]            # decoded text of each token ('' for special tokens)
    toklen: np.ndarray        # int32: len(txt[i])
    has_nl: np.ndarray        # bool: token contains a newline
    pre_nl_len: np.ndarray    # int32: chars before the first newline (== toklen if none)
    firstchar: np.ndarray     # '<U1': first character ('' for empty)
    first_upper: np.ndarray   # bool: first *alphabetic* char is uppercase
    first_lower: np.ndarray   # bool: first *alphabetic* char is lowercase
    has_banned: np.ndarray    # bool: token carries a banned formatting char
    eos_id: int

    @classmethod
    def build(cls, llm) -> "_TokenTable":
        n = llm.n_vocab()
        txt = [llm.detokenize([i], special=False).decode("utf-8", "replace") for i in range(n)]
        toklen = np.fromiter((len(s) for s in txt), dtype=np.int32, count=n)
        has_nl = np.fromiter(("\n" in s for s in txt), dtype=bool, count=n)
        pre_nl_len = np.fromiter(
            ((s.index("\n") if "\n" in s else len(s)) for s in txt), dtype=np.int32, count=n
        )
        firstchar = np.array([s[0] if s else "" for s in txt], dtype="<U1")

        def _first_alpha_case(s: str) -> int:
            for ch in s:
                if ch.isalpha():
                    return 1 if ch.isupper() else -1 if ch.islower() else 0
            return 0

        cases = np.fromiter((_first_alpha_case(s) for s in txt), dtype=np.int8, count=n)
        has_banned = np.fromiter(
            (any(c in _BANNED_CHARS for c in s) for s in txt), dtype=bool, count=n
        )
        return cls(
            txt=txt, toklen=toklen, has_nl=has_nl, pre_nl_len=pre_nl_len,
            firstchar=firstchar, first_upper=(cases == 1), first_lower=(cases == -1),
            has_banned=has_banned, eos_id=llm.token_eos(),
        )


class _LineMask:
    """A llama-cpp logits_processor: masks one acrostic line and records, per
    step, the log-prob of the (greedy) chosen token — the signal the crossing
    search scores. One instance per generated line/rollout.

    The grammar for a single line is just: forced letter, then up to
    `max_line` total chars, ending in a newline (or EOS on the last line). So
    the mask is computed directly rather than via an NFA walk:
      * first step  (line empty) → tokens whose first char is the forced letter
                                   (case-insensitive) and that carry no newline;
      * later steps (in body)    → non-newline tokens that fit the char budget,
                                   plus newline enders once the line is long
                                   enough (or EOS, on the last line).
    """

    def __init__(self, table: _TokenTable, llm, allowed_first: set[str], *,
                 min_line: int, max_line: int, is_last: bool, mid: bool,
                 want_logps: bool = True):
        self.t = table
        self.llm = llm
        self.allowed_first = allowed_first
        self.min_line = min_line
        self.max_line = max_line
        self.is_last = is_last
        self.mid = mid
        self.want_logps = want_logps     # only the crossing search reads logps
        self.prompt_len: int | None = None
        self.logps: list[float] = []     # chosen (argmax-legal) log-prob, per step

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if self.prompt_len is None:
            self.prompt_len = len(input_ids)
        gen_ids = input_ids[self.prompt_len:]
        gen = self.llm.detokenize([int(i) for i in gen_ids]).decode("utf-8", "replace") if len(gen_ids) else ""
        line_len = len(gen)
        t = self.t

        if line_len == 0:
            # Forced-letter step: first char in the allowed set, no newline.
            legal = np.isin(t.firstchar, list(self.allowed_first)) & ~t.has_nl
            if self.mid and (legal & t.first_lower).any():
                # Stealth: prefer the lowercase forced letter mid-sentence.
                legal &= ~t.first_upper
        else:
            remaining = self.max_line - line_len
            legal = (~t.has_nl) & (t.toklen > 0) & (t.toklen <= remaining)
            if line_len >= self.min_line:
                if self.is_last:
                    legal = legal.copy()
                    legal[t.eos_id] = True
                else:
                    legal |= t.has_nl & (t.pre_nl_len <= remaining)

        legal &= ~t.has_banned   # plain prose only — no markdown/backslash tokens

        # Record the chosen-token log-prob (greedy → argmax over legal logits)
        # under the masked, renormalised distribution. Only the crossing search
        # reads this, so greedy skips the logsumexp over the (huge) legal set.
        if self.want_logps:
            legal_logits = scores[legal]
            if legal_logits.size:
                m = float(legal_logits.max())
                lse = m + float(np.log(np.exp(legal_logits - m).sum()))
                self.logps.append(m - lse)
            else:
                self.logps.append(0.0)
        scores[~legal] = -np.inf
        return scores


class _NewlineStop:
    """Stop a line as soon as the freshly sampled token carries a newline."""

    def __init__(self, table: _TokenTable, prompt_len: int):
        self.t = table
        self.prompt_len = prompt_len

    def __call__(self, input_ids: np.ndarray, logits: np.ndarray) -> bool:
        if len(input_ids) <= self.prompt_len:
            return False
        return "\n" in self.t.txt[int(input_ids[-1])]


# Default acrostic: three paragraphs spelling ABCD / EFGH / ABCDEFGHIJKL.
DEFAULT_PARAGRAPHS = ("ABCD", "EFGH", "ABCDEFGHIJKL")


class AcrosticDecoder:
    """Generates a multi-paragraph acrostic from a chat prompt, one line at a
    time, under the grammar mask and (optionally) the local-crossing search.

    Public-API only: each line/rollout is a fresh `create_completion` whose
    prompt is the chat-templated system+user plus everything committed so far,
    re-fed as text. llama-cpp's prefix KV-cache makes the shared prefix cheap.
    """

    def __init__(self, llm, *, min_line: int = 60, max_line: int = 80,
                 search: bool = True, k: int = 4, j: int = 3, R: int = 4,
                 repeat_penalty: float = 1.15):
        self.llm = llm
        self.min_line = min_line
        self.max_line = max_line
        self.search = search
        self.k, self.j, self.R = k, j, R
        self.repeat_penalty = repeat_penalty
        self.table = _TokenTable.build(llm)
        self._formatter = None  # lazy chat formatter

    # -- prompt plumbing -------------------------------------------------
    def _render_prompt(self, system: str, user: str) -> str:
        if self._formatter is None:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            md = self.llm.metadata
            self._formatter = Jinja2ChatFormatter(
                template=md["tokenizer.chat_template"],
                eos_token="<|im_end|>", bos_token="", add_generation_prompt=True,
            )
        res = self._formatter(messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return res.prompt

    def _ids(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=True, special=True)

    # -- generation primitives ------------------------------------------
    def _gen_line(self, base: str, committed: str, letter: str, is_last: bool):
        """Greedily generate one line (forced first `letter`) continuing `base +
        committed`. Returns (line_text incl. trailing newline, per-token logps)."""
        prompt_ids = self._ids(base + committed)
        mask = _LineMask(
            self.table, self.llm, _letter_set(letter),
            min_line=self.min_line, max_line=self.max_line,
            is_last=is_last, mid=mid_sentence(committed), want_logps=self.search,
        )
        from llama_cpp import LogitsProcessorList, StoppingCriteriaList

        stops = None if is_last else StoppingCriteriaList([_NewlineStop(self.table, len(prompt_ids))])
        out = self.llm.create_completion(
            prompt=prompt_ids, max_tokens=self.max_line + 8,
            temperature=0.0, top_k=0, top_p=1.0, repeat_penalty=self.repeat_penalty,
            logits_processor=LogitsProcessorList([mask]), stopping_criteria=stops,
        )
        text = out["choices"][0]["text"]
        if not is_last:
            nl = text.find("\n")
            if nl != -1:
                text = text[:nl + 1]
        return text, mask.logps

    def _roll_open(self, base: str, committed: str, letter: str, n: int) -> list[float]:
        """Roll the next line's opening — forced `letter` + up to n-1 tokens —
        and return their log-probs (the crossing search's 'after' window)."""
        prompt_ids = self._ids(base + committed)
        mask = _LineMask(
            self.table, self.llm, _letter_set(letter),
            min_line=self.min_line, max_line=self.max_line,
            is_last=False, mid=mid_sentence(committed),
        )
        from llama_cpp import LogitsProcessorList, StoppingCriteriaList

        self.llm.create_completion(
            prompt=prompt_ids, max_tokens=n,
            temperature=0.0, top_k=0, top_p=1.0, repeat_penalty=self.repeat_penalty,
            logits_processor=LogitsProcessorList([mask]),
            stopping_criteria=StoppingCriteriaList([_NewlineStop(self.table, len(prompt_ids))]),
        )
        return mask.logps

    def _search_break(self, base: str, committed: str, text: str, logps: list[float],
                      sep: str, next_letter: str) -> str:
        """Local-crossing search: pick where to break this line. `text` is the
        greedy line (ending in '\\n'); returns the chosen prefix + `sep`."""
        line_ids = self._ids(base + committed + text)[len(self._ids(base + committed)):]
        m = min(len(line_ids), len(logps))
        if m == 0:
            return text[:-1] + sep if text.endswith("\n") else text + sep
        ids, lps = line_ids[-m:], logps[-m:]
        has_nl = text.endswith("\n")

        candidates: list[tuple[float, str]] = []
        for r in range(0, min(self.R, m - 1) + 1):
            if r > 0:
                ft = self.table.txt[ids[m - r]]
                if not ft or not _BREAK_TOKEN.match(ft):
                    continue
            kept = ids[:m - r]
            if not kept:
                continue
            prefix = self.llm.detokenize(kept).decode("utf-8", "replace").rstrip("\n")
            if r > 0 and len(prefix) < self.min_line:
                continue
            before = lps[:m - r]
            if r == 0 and has_nl:
                before = before[:-1]
            before = before[-self.k:]
            after = self._roll_open(base, committed + prefix + sep, next_letter, 1 + self.j)
            window = before + after
            score = sum(window) / len(window) if window else 0.0
            candidates.append((score, prefix + sep))

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            return candidates[0][1]
        return (text[:-1] if has_nl else text) + sep

    # -- public API ------------------------------------------------------
    def generate(self, system: str, user: str,
                 paragraphs: tuple[str, ...] = DEFAULT_PARAGRAPHS) -> str:
        base = self._render_prompt(system, user)

        # Flatten paragraphs into (forced letter, separator-after-this-line).
        plan: list[tuple[str, str]] = []
        for p, secret in enumerate(paragraphs):
            for i, letter in enumerate(secret):
                last_overall = (p == len(paragraphs) - 1) and (i == len(secret) - 1)
                if last_overall:
                    sep = ""
                elif i == len(secret) - 1:
                    sep = "\n\n"   # paragraph break
                else:
                    sep = "\n"     # line break within a paragraph
                plan.append((letter, sep))

        committed = ""
        for idx, (letter, sep) in enumerate(plan):
            is_last = idx == len(plan) - 1
            text, logps = self._gen_line(base, committed, letter, is_last)
            if is_last:
                committed += text.rstrip()
                break
            if self.search:
                next_letter = plan[idx + 1][0]
                committed += self._search_break(base, committed, text, logps, sep, next_letter)
            else:
                committed += (text[:-1] if text.endswith("\n") else text) + sep
        return _trim_last_sentence(committed.strip())


def _letter_set(letter: str) -> set[str]:
    """Allowed first characters for a forced letter — both cases (the mask's
    stealth step decides which to keep)."""
    return {letter.upper(), letter.lower()}


def _trim_last_sentence(text: str) -> str:
    """The last line stops at the char wall (no newline to break on), often mid-
    word. Trim only that line back to its final sentence-ending punctuation for a
    clean close; leave it alone if it has none (so the forced letter survives)."""
    lines = text.split("\n")
    last = lines[-1]
    cut = max(last.rfind("."), last.rfind("!"), last.rfind("?"))
    if cut != -1 and cut < len(last) - 1:
        lines[-1] = last[:cut + 1]
        return "\n".join(lines)
    return text
