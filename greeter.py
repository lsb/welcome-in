"""Stage 2 — the greeting.

A small Qwen3.5 (GGUF, CPU via llama-cpp-python) speaks as the voice of the
gallery doorway. The model is loaded lazily so nothing heavy happens until a
visitor actually faces the camera.
"""

from __future__ import annotations

import re

from face_gate import FaceResult
from models import ensure_gguf

SYSTEM_PROMPT = (
    "You are the voice of 'Welcome In', a gentle, slightly poetic presence at the "
    "doorway of an art gallery. When a visitor turns to look at you, you welcome them "
    "with one or two short, warm, original sentences — present, unhurried, and sincere. "
    "Never use emojis, hashtags, quotation marks, or stage directions, and never repeat "
    "an earlier greeting. /no_think"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class Greeter:
    def __init__(self, size: str = "0.8B", n_ctx: int = 2048, n_threads: int | None = None):
        self.size = size
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None  # lazy

    def _ensure_llm(self):
        if self._llm is None:
            from llama_cpp import Llama

            path = ensure_gguf(self.size)
            print(f"[greeter] loading Qwen3.5-{self.size} (CPU) …")
            self._llm = Llama(
                model_path=str(path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._llm

    def greet(self, face: FaceResult) -> str:
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
