"""Stage 2 — the greeting.

A small Qwen3.5 (GGUF, CPU via llama-cpp-python) speaks as the voice of the
gallery doorway. The model is loaded lazily so nothing heavy happens until a
visitor actually faces the camera.

Two styles:
  * "short"    — one or two warm sentences (the original behaviour).
  * "acrostic" — a long, effusive welcome whose lines secretly spell an acrostic,
                 decoded under a grammar mask + local-crossing search (acrostic.py,
                 ported from github.com/lsb/sidechat).
"""

from __future__ import annotations

import re

from acrostic import DEFAULT_PARAGRAPHS
from face_gate import FaceResult
from models import ensure_gguf

SYSTEM_PROMPT = (
    "You are the voice of 'Welcome In', a gentle, slightly poetic presence at the "
    "doorway of an art gallery. When a visitor turns to look at you, you welcome them "
    "with one or two short, warm, original sentences — present, unhurried, and sincere. "
    "Never use emojis, hashtags, quotation marks, or stage directions, and never repeat "
    "an earlier greeting. /no_think"
)

# Effusive style. Plain-prose instruction matters: the per-line acrostic chopper
# shreds markdown into fragments, so we ask for flowing prose (sidechat's
# "markdown suppression" lesson; the mask enforces it too).
ACROSTIC_SYSTEM = (
    "You are the voice of 'Welcome In', a warm, openhearted, effusive presence at the "
    "doorway of an art gallery. When a visitor turns to look at you, you pour out a long, "
    "generous, heartfelt welcome — vivid, sincere, and overflowing with delight that they "
    "have arrived and with wonder at the art that awaits them. Write three flowing "
    "paragraphs of plain prose. Never use lists, bullet points, headings, markdown, "
    "emojis, hashtags, quotation marks, or stage directions. /no_think"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class Greeter:
    def __init__(self, size: str = "0.8B", n_ctx: int = 2048, n_threads: int | None = None,
                 style: str = "acrostic", search: bool = False, search_R: int = 4,
                 paragraphs: tuple[str, ...] = DEFAULT_PARAGRAPHS):
        self.size = size
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.style = style
        self.search = search
        self.search_R = search_R
        self.paragraphs = paragraphs
        self._llm = None       # lazy
        self._decoder = None   # lazy (acrostic style only)

    def load(self) -> "Greeter":
        """Eagerly load the model (so callers can time it). Returns self."""
        self._ensure_llm()
        if self.style == "acrostic":
            self._ensure_decoder()  # build the token table up front, too
        return self

    def _ensure_llm(self):
        if self._llm is None:
            from llama_cpp import Llama

            path = ensure_gguf(self.size)
            self._llm = Llama(
                model_path=str(path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._llm

    def _ensure_decoder(self):
        if self._decoder is None:
            from acrostic import AcrosticDecoder

            self._decoder = AcrosticDecoder(
                self._ensure_llm(), search=self.search, R=self.search_R,
            )
        return self._decoder

    def greet(self, face: FaceResult) -> str:
        if self.style == "acrostic":
            return self._greet_acrostic(face)
        return self._greet_short(face)

    # -- styles ----------------------------------------------------------
    def _greet_acrostic(self, face: FaceResult) -> str:
        dec = self._ensure_decoder()
        user = (
            f"A visitor has just turned to face you — {face.description}. "
            "Pour out a long, effusive, heartfelt welcome into the gallery."
        )
        return dec.generate(ACROSTIC_SYSTEM, user, self.paragraphs)

    def _greet_short(self, face: FaceResult) -> str:
        llm = self._ensure_llm()
        user = (
            f"A visitor has just turned to face you — {face.description}. "
            "Offer them a warm welcome into the gallery."
        )
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.8,
            top_p=0.9,
            max_tokens=64,
            repeat_penalty=1.1,
        )
        text = resp["choices"][0]["message"]["content"]
        return self._clean(text)

    @staticmethod
    def _clean(text: str) -> str:
        text = _THINK_RE.sub("", text)
        # drop any stray opening <think> with no close
        text = text.split("</think>")[-1]
        text = text.replace("<think>", "")
        return text.strip().strip('"').strip()
