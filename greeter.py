"""Stage 2 — the greeting, in two parts.

A small Qwen3.5 (GGUF, CPU via llama-cpp-python) speaks as the gallery's voice.
Each visitor gets two things, generated as two independent chats:

  Part 1 — the hello: a short, casual, spoken welcome that notes what the visitor
    is wearing. Generated RAW (no grammar mask), because the terse voice we want is
    a few plain sentences and the acrostic's line quota only fights that (see the
    SYSTEM comment below).
  Part 2 — the question card: a few short, open questions about generative AI, on
    one of seventeen randomly chosen themes (questions.py), so the visitor argues and
    wonders for themselves rather than watching passively. Generated UNDER the acrostic
    mask (acrostic.py; grammar idea from github.com/lsb/sidechat) — its length suits a
    handful of questions, and the acrostic lives here now instead of in the hello.

The model is loaded lazily so nothing heavy happens until a visitor faces the
camera; both parts share the one loaded model.
"""

from __future__ import annotations

from dataclasses import dataclass

from acrostic import acrostic_from_env
from face_gate import FaceResult
from models import ensure_gguf
from questions import QUESTIONS_SYSTEM, build_questions_prompt, pick_topic

# Part 1 register: a short, casual greeting — two or three plain sentences (a quick
# hi + one compliment, the gallery's name, one practical line), not a speech. Two
# example greetings carry that register far more reliably than any description; their
# clothing (corduroy jacket, scarf) is deliberately nothing a real visitor here
# wears, so any bleed into the output is obvious. Part 1 is always generated raw (no
# acrostic), so the terse voice is never fought by the line quota — the acrostic now
# lives in the Part 2 question card instead. Best on the 2B model; a 0.8B is
# likelier to parrot an example outfit.
SYSTEM = (
    "You are the host at the door of The Intersection of Art and Technology, a gallery "
    "where art and technology meet. A visitor just walked in. Greet them out loud the way "
    "a warm, easygoing host actually talks: two or three short sentences, no more. "
    "Open with a quick hi and one genuine compliment on the most distinctive thing they "
    "are wearing. Welcome them to the intersection of art and technology. Then close with "
    "one warm, practical line and stop: invite them to browse as long as they like and "
    "mention that a lot of the work is for sale. The whole greeting is three short "
    "sentences, nothing more. Keep it plain and friendly — a hello, not a speech. Do not "
    "invent details such as prices, discounts, or websites. No gushing, no flourishes, no "
    "questions, no exclamation marks, no quotation marks.\n\n"
    "Examples of the right length and tone:\n"
    "Hi, love the corduroy jacket. Welcome to the intersection of art and technology. "
    "Stay as long as you like, and a lot of the work here is for sale.\n"
    "Hi, great scarf. This is the intersection of art and technology; thanks for coming "
    "in. Browse as long as you want, and a lot of these pieces are for sale. /no_think"
)

@dataclass
class Greeting:
    """What a visitor gets: the spoken hello (Part 1) and the acrostic question
    card (Part 2) on a randomly chosen theme."""
    hello: str
    questions: str
    topic: str


class Greeter:
    def __init__(self, size: str = "0.8B", n_ctx: int = 2048, n_threads: int | None = None,
                 paragraphs: tuple[str, ...] | None = None,
                 min_line: int | None = None, max_line: int | None = None):
        self.size = size
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        # The acrostic constraint (used for the Part 2 question card) comes from the
        # environment (WELCOME_ACROSTIC / WELCOME_MIN_LINE / WELCOME_MAX_LINE) so it
        # can be tuned without code edits; an explicit argument still wins.
        env_paragraphs, env_min, env_max = acrostic_from_env()
        self.paragraphs = env_paragraphs if paragraphs is None else paragraphs
        self.min_line = env_min if min_line is None else min_line
        self.max_line = env_max if max_line is None else max_line
        self._llm = None         # lazy
        self._hello_dec = None   # lazy: Part 1 hello, raw (no grammar)
        self._card_dec = None    # lazy: Part 2 question card, acrostic-masked

    def load(self) -> "Greeter":
        """Eagerly load the model + compile the grammar (so callers can time it)."""
        self._ensure_decoders()
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

    def _ensure_decoders(self):
        if self._hello_dec is None:
            from acrostic import AcrosticDecoder

            llm = self._ensure_llm()
            # Part 1 hello: paragraphs=() -> no grammar mask, so the terse voice is
            # never forced to fill lines. Part 2 card: the configured acrostic.
            self._hello_dec = AcrosticDecoder(llm, paragraphs=())
            self._card_dec = AcrosticDecoder(llm, paragraphs=self.paragraphs,
                                             min_line=self.min_line, max_line=self.max_line)
        return self._hello_dec, self._card_dec

    def greet(self, face: FaceResult, clothing: str | None = None,
              on_delta=None, abort=None) -> Greeting:
        """Generate both parts. If on_delta is given it is called as
        on_delta(part, piece) for every streamed token — part is "hello" while
        the spoken hello streams, then "card" while the question card streams —
        so a live display can render the greeting as the model writes it. Other
        sinks (the printer) use the finished, settled text in the returned
        Greeting, not the raw stream.

        ``abort`` is forwarded to both streamed parts (see AcrosticDecoder.generate):
        when a group's worth of visitors all leave mid-greeting it raises
        GenerationAborted, which the kiosk catches to drop the half-written card.

        How many faces are turned to the camera (``face.facing_count``) decides
        whether the hello is addressed to one visitor or to the group in the plural."""
        hello_dec, card_dec = self._ensure_decoders()
        group_size = max(1, getattr(face, "facing_count", 1) or 1)
        hello = hello_dec.generate(
            SYSTEM, build_user_prompt(clothing, group_size),
            on_delta=(lambda d: on_delta("hello", d)) if on_delta else None, abort=abort)
        topic = pick_topic()
        questions = card_dec.generate(
            QUESTIONS_SYSTEM, build_questions_prompt(topic),
            on_delta=(lambda d: on_delta("card", d)) if on_delta else None, abort=abort)
        return Greeting(hello=hello, questions=questions, topic=topic)


def build_user_prompt(clothing: str | None, group_size: int = 1) -> str:
    """The per-visitor user turn — deliberately minimal so the system prompt's short,
    casual register carries the greeting. Kept module-level so the tone harness and
    the live greeter build the exact same prompt (no drift between test and ship).

    For a group (group_size >= 2) the turn names the headcount and asks for a single
    plural hello to everyone, with the one outfit we read (the closest visitor's)
    used as the compliment — so 'love the jacket' lands on a person, not the crowd."""
    if group_size >= 2:
        if clothing:
            return (f"A group of {group_size} visitors just walked in together. "
                    f"The one nearest you is wearing {clothing}. Greet them all at once: "
                    f"one quick hi to the group, then one compliment on that outfit. "
                    f"Speak to the whole group, not one person.")
        return (f"A group of {group_size} visitors just walked in together. "
                f"Greet them all at once with one quick hi to the group. "
                f"Speak to the whole group, not one person.")
    if clothing:
        return f"A visitor just walked in, wearing {clothing}. Greet them."
    return "A visitor just walked in. Greet them."
