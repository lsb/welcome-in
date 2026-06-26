"""Building one question card from several acrostic stanzas.

A card is N independent decode runs for the *same* question — one per acrostic in
the configured set (default SLOP / TIAT / LEEB) — joined into one body. Each run
is its own short grammar-masked pass, which stays crisp where a single long
acrostic would pad and degenerate. Shared here so the warmup (0.8B) and the
background producer (27B) build cards identically and the printer/screen see the
same shape regardless of which model wrote them.
"""

from __future__ import annotations

from acrostic import (AcrosticDecoder, DEFAULT_MAX_LINE, DEFAULT_MIN_LINE,
                      GenerationAborted)
from acrostics import AcrosticSpec
from models import ensure_gguf
from questions import QUESTIONS_SYSTEM, build_questions_prompt

# Stanzas are joined with a blank line; both the screen label and the printer
# (card_postscript advances a line on each blank) already render that as a break.
STANZA_SEP = "\n\n"


def make_llm(size: str, n_ctx: int = 2048, n_threads: int | None = None):
    """Load a GGUF greeter. mmap (not mlock) so weight pages stay file-backed and
    evictable — important when the 27B (~13.6 GB) is co-resident on a 16 GB Pi."""
    from llama_cpp import Llama

    return Llama(model_path=str(ensure_gguf(size)), n_ctx=n_ctx,
                 n_threads=n_threads, use_mmap=True, use_mlock=False, verbose=False)


def build_decoders(llm, acrostics: tuple[AcrosticSpec, ...], *,
                   no_think: bool = False) -> dict[str, AcrosticDecoder]:
    """One AcrosticDecoder per acrostic spec (each compiles its grammar once), all
    sharing the single llm. Pass ``no_think=True`` for a reasoning model (27B)."""
    return {
        spec.secret: AcrosticDecoder(
            llm, paragraphs=(spec.secret,),
            min_line=spec.min_line if spec.min_line is not None else DEFAULT_MIN_LINE,
            max_line=spec.max_line if spec.max_line is not None else DEFAULT_MAX_LINE,
            no_think=no_think)
        for spec in acrostics
    }


def synthesize_card(decoders: dict[str, AcrosticDecoder],
                    acrostics: tuple[AcrosticSpec, ...], topic: str, *,
                    seed_base: int, abort=None) -> str:
    """Generate one card: an independent run per acrostic, joined into one body.

    ``seed_base`` is offset per stanza (``seed_base + i``) so the stanzas diverge
    and successive cards for a topic stay varied — we never rely on the decoder's
    time-based default seed, which would collide across runs in the same second.
    When ``abort`` is given the streaming (abortable) decode path is used so a
    multi-minute 27B run tears down promptly at shutdown.
    """
    on_delta = (lambda d: None) if abort is not None else None
    stanzas = [
        decoders[spec.secret].generate(
            QUESTIONS_SYSTEM, build_questions_prompt(topic),
            seed=seed_base + i, on_delta=on_delta, abort=abort)
        for i, spec in enumerate(acrostics)
    ]
    return STANZA_SEP.join(stanzas)


__all__ = ["make_llm", "build_decoders", "synthesize_card", "GenerationAborted",
           "STANZA_SEP"]
