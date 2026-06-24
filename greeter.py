"""Stage 2 — the greeting.

A small Qwen3.5 (GGUF, CPU via llama-cpp-python) speaks as the gallery's voice.
It generates a long, effusive welcome whose lines secretly spell an acrostic,
decoded under a GBNF grammar mask (acrostic.py; the grammar idea from
github.com/lsb/sidechat). The model is loaded lazily so nothing heavy happens
until a visitor actually faces the camera.
"""

from __future__ import annotations

from acrostic import DEFAULT_PARAGRAPHS
from face_gate import FaceResult
from models import ensure_gguf

# The {works}/{n} placeholders are filled per greeting. Plain-prose instruction
# matters: the per-line acrostic chopper shreds markdown, so we ask for flowing
# prose (the mask enforces it too).
SYSTEM = (
    "You are the professional, warm, crisp, bright voice of an art gallery at the "
    "intersection of art and technology. Speak directly to the visitor in the second person "
    "('you', 'your') as a welcome addressed straight to them — never describe them in the "
    "third person (no 'the visitor', no 'they'). Efficiently note a couple of specific "
    "details about what you can see them wearing (observe plainly, do not gush), tell them "
    "they are entering a gallery at the intersection of art and technology, and mention that "
    "a lot of the art is for sale. Write a single flowing paragraph of plain prose, "
    "professional and efficient. Never use lists, bullet points, headings, markdown, emojis, "
    "hashtags, quotation marks, or stage directions. /no_think"
)

class Greeter:
    # Spell the count out (the model renders a bare digit like 47 as "four seven").
    def __init__(self, size: str = "0.8B", n_ctx: int = 2048, n_threads: int | None = None,
                 paragraphs: tuple[str, ...] = DEFAULT_PARAGRAPHS):
        self.size = size
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.paragraphs = paragraphs
        self._llm = None       # lazy
        self._decoder = None   # lazy

    def load(self) -> "Greeter":
        """Eagerly load the model + compile the grammar (so callers can time it)."""
        self._ensure_decoder()
        return self

    def _ensure_llm(self):
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(ensure_gguf(self.size)),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._llm

    def _ensure_decoder(self):
        if self._decoder is None:
            from acrostic import AcrosticDecoder

            self._decoder = AcrosticDecoder(self._ensure_llm(), paragraphs=self.paragraphs)
        return self._decoder

    def greet(self, face: FaceResult, clothing: str | None = None) -> str:
        dec = self._ensure_decoder()
        system = SYSTEM
        if clothing:
            user = (
                "Greet the person now in front of the camera, speaking straight to them as "
                f"'you'. You can see they are wearing {clothing}. Note a detail or two about "
                "it, welcome them into the gallery, and mention that much of the art is for sale."
            )
        else:
            user = (
                "Greet the person now in front of the camera, speaking straight to them as "
                "'you'. Welcome them into the gallery and mention that much of the art is for sale."
            )
        return dec.generate(system, user)
