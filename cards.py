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

# --- prompt-bleed gate ------------------------------------------------------
# Even with reasoning off, the forced first letter of a stanza (especially the
# TIAT line, T → "The…") sometimes pulls the 27B into describing its task instead
# of asking questions: "The prompt asks me to write a question card…", then it
# regurgitates the brief. These phrases describe the task and essentially never
# occur in a genuine visitor question, so a stanza containing one is regenerated
# with a fresh seed. Bare "prompt"/"write"/"let's" are deliberately NOT here, so a
# real question like "Should we judge the art or the prompt?" passes.
_META_LEAK = (
    "the prompt asks", "question card", "pull visitors", "turn visitors",
    "make them argue", "no preamble", "no headings", "no answers allowed",
    "rooted in everyday", "plain, direct, rooted", "based on the theme",
    "based on this theme", "the theme:", "the theme is", "the theme of",
    "the tone should", "tone should", "the tone:", "tone:", "i need to write",
    "i must write", "i should write", "i need to follow", "i'm looking for",
    "i was asked", "i have to write", "art gallery featuring", "in an art gallery",
    "art gallery full of", "probing questions", "short, open questions",
    "open questions that", "open-ended questions", "please provide", "provide me with",
    "the user wants", "want me to", "wants me to", "questions for visitors",
    "for visitors", "a handful of", "specific instructions", "following the theme",
    "for the theme",
)
MAX_STANZA_TRIES = 8


def meta_leak(text: str) -> str | None:
    """The offending phrase if a stanza describes its task instead of asking
    questions (prompt bleed), else None."""
    low = text.lower()
    return next((p for p in _META_LEAK if p in low), None)


class CardRejected(Exception):
    """A stanza leaked the prompt on every retry; the caller drops the card."""


def make_llm(size: str, n_ctx: int = 2048, n_threads: int | None = None):
    """Load a GGUF greeter. mmap (not mlock) so weight pages stay file-backed and
    evictable — important when the 27B (~13.6 GB) is co-resident on a 16 GB Pi."""
    from llama_cpp import Llama

    return Llama(model_path=str(ensure_gguf(size)), n_ctx=n_ctx,
                 n_threads=n_threads, use_mmap=True, use_mlock=False, verbose=False)


def build_decoders(llm, acrostics: tuple[AcrosticSpec, ...], *,
                   no_think: bool = False,
                   temperature: float | None = None) -> dict[str, AcrosticDecoder]:
    """One AcrosticDecoder per acrostic spec (each compiles its grammar once), all
    sharing the single llm. ``no_think=True`` for a reasoning model (27B);
    ``temperature`` overrides the decoder default (higher = more varied cards)."""
    temp_kw = {} if temperature is None else {"temperature": temperature}
    return {
        spec.secret: AcrosticDecoder(
            llm, paragraphs=(spec.secret,),
            min_line=spec.min_line if spec.min_line is not None else DEFAULT_MIN_LINE,
            max_line=spec.max_line if spec.max_line is not None else DEFAULT_MAX_LINE,
            no_think=no_think, **temp_kw)
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
    multi-minute 27B run tears down promptly at shutdown. Any stanza that leaks the
    prompt (see ``meta_leak``) is regenerated with a fresh seed up to
    ``MAX_STANZA_TRIES``; if it never comes clean the card is dropped (``CardRejected``).
    """
    on_delta = (lambda d: None) if abort is not None else None
    stanzas = []
    for i, spec in enumerate(acrostics):
        dec = decoders[spec.secret]
        for attempt in range(MAX_STANZA_TRIES):
            text = dec.generate(
                QUESTIONS_SYSTEM, build_questions_prompt(topic),
                seed=seed_base + i * 101 + attempt * 9973, on_delta=on_delta, abort=abort)
            leak = meta_leak(text)
            if leak is None:
                break
            print(f"[cards] {spec.secret} stanza leaked {leak!r}; regenerating "
                  f"(attempt {attempt + 1}/{MAX_STANZA_TRIES})")
        else:
            raise CardRejected(f"{spec.secret} stanza kept leaking for {topic!r}")
        stanzas.append(text)
    return STANZA_SEP.join(stanzas)


__all__ = ["make_llm", "build_decoders", "synthesize_card", "meta_leak",
           "CardRejected", "GenerationAborted", "STANZA_SEP"]
