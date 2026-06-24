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
    "You are the bright, enthusiastic, professional voice of a contemporary art gallery at "
    "the intersection of art and technology. When a visitor steps up to the camera, you "
    "greet them with genuine warmth and excitement: pay them a sincere compliment on how "
    "wonderful they look on the webcam right now, welcome them to the gallery, and let them "
    "know that {works} works are currently on display and for sale. Write {n} flowing "
    "paragraphs of upbeat, polished plain prose. Never use lists, bullet points, headings, "
    "markdown, emojis, hashtags, quotation marks, or stage directions. /no_think"
)

class Greeter:
    # Spell the count out (the model renders a bare digit like 47 as "four seven").
    def __init__(self, size: str = "0.8B", n_ctx: int = 2048, n_threads: int | None = None,
                 paragraphs: tuple[str, ...] = DEFAULT_PARAGRAPHS,
                 works_for_sale: str = "forty-seven"):
        self.size = size
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.paragraphs = paragraphs
        self.works_for_sale = works_for_sale
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

    def greet(self, face: FaceResult) -> str:
        dec = self._ensure_decoder()
        system = SYSTEM.format(n=len(self.paragraphs), works=self.works_for_sale)
        user = (
            f"A visitor has just stepped up to the camera — {face.description}. "
            "Welcome them now."
        )
        return dec.generate(system, user)
