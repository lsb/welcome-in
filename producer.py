"""The card producer: a background worker that pre-fills the pool.

Runs the heavy model (Qwen3.6 27B Q3_K_M, reasoning off) and keeps generating
question cards into the on-disk pool (pool.py) until every question has ``cap``
cards, then idles and tops up after serves/evictions. Serving never waits on it —
the kiosk reads finished cards from the pool and types them out.

When an idle pool is wired in, each card's first stanza — itself a complete
single-acrostic (SLOP) stanza — is also shed into it, so the standing screen's
ambient rotation refreshes off the same 27B run with no extra inference.

Two ways to run it:
  * In-process, as a daemon thread the kiosk starts after warmup (live top-up).
  * Standalone (``python producer.py …``), to pre-fill a pool offline on a faster
    box and commit/ship the resulting card files. See ``main``.

A 27B card is minutes/run on CPU — that's fine; this is background work, decoupled
from the instant serve path. The producer is abortable: a multi-minute decode tears
down at shutdown because ``synthesize_card`` is given the streaming abort hook.
"""

from __future__ import annotations

import argparse
import threading
import time

from acrostics import AcrosticSpec, acrostics_signature, load_acrostics
from cards import (STANZA_SEP, CardRejected, GenerationAborted, build_decoders,
                   make_llm, synthesize_card)
from pool import CardPool
from questions import TOPICS


# Qwen's recommended sampling temperature for non-thinking mode (which is how the
# producer runs — reasoning off). Higher than the tiny-model greeter's tuned 0.3;
# this is an art installation, so we lean into the extra variety.
PRODUCER_TEMPERATURE = 0.7


class PoolProducer:
    def __init__(self, pool: CardPool, acrostics: tuple[AcrosticSpec, ...],
                 model: str = "27B", topics: tuple[str, ...] = TOPICS,
                 n_threads: int | None = None, no_think: bool = True,
                 cap: int | None = None, idle_sleep: float = 10.0,
                 idle_pool: CardPool | None = None):
        self.pool = pool
        self.acrostics = acrostics
        self.model = model
        self.topics = topics
        self.n_threads = n_threads
        self.no_think = no_think
        self.cap = cap if cap is not None else pool.cap
        self.idle_sleep = idle_sleep
        # Optional companion pool for the standing screen's ambient rotation. A main
        # card's first stanza is itself a complete single-acrostic (SLOP) stanza, so
        # we shed it into the idle pool as each card is made — the idle text refreshes
        # off the same 27B run that fills the main pool, with no extra inference.
        self.idle_pool = idle_pool
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._decoders: dict | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "PoolProducer":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- model + work ----------------------------------------------------
    def _load(self) -> None:
        self._llm = make_llm(self.model, n_threads=self.n_threads)
        self._decoders = build_decoders(self._llm, self.acrostics,
                                        no_think=self.no_think,
                                        temperature=PRODUCER_TEMPERATURE)

    def _next_topic(self) -> str | None:
        """The question most in need of cards: lowest real-card count, under cap.
        Round-robins toward ``cap`` per topic; ``None`` when every topic is full."""
        counts = self.pool.counts_by_topic()
        under = sorted((counts.get(t, 0), i, t)
                       for i, t in enumerate(self.topics)
                       if counts.get(t, 0) < self.cap)
        return under[0][2] if under else None

    def make_card(self, topic: str) -> str:
        return synthesize_card(self._decoders, self.acrostics, topic,
                               seed_base=time.time_ns() & 0x7FFFFFFF,
                               abort=self._stop.is_set)

    def shed_idle(self, topic: str, text: str) -> None:
        """Mirror a freshly made card's first stanza — already a complete
        single-acrostic stanza — into the idle pool, honouring its own per-topic cap.
        No-op when no idle pool is wired. The first stanza spells ``acrostics[0]``,
        which is exactly the acrostic the idle pool is scoped to, so they stay in
        sync if the acrostic set ever changes."""
        if self.idle_pool is None:
            return
        stanza = text.split(STANZA_SEP, 1)[0].strip()
        if stanza:
            self.idle_pool.insert_card(topic, stanza, model=self.model,
                                       provisional=False)

    def _run(self) -> None:
        try:
            self._load()
        except Exception as e:  # never take down the kiosk over a producer failure
            print(f"[producer] load failed: {e}")
            return
        print(f"[producer] {self.model} ready; filling pool to {self.cap}/topic")
        while not self._stop.is_set():
            topic = self._next_topic()
            if topic is None:                 # pool full -> idle, top up later
                self._stop.wait(self.idle_sleep)
                continue
            try:
                text = self.make_card(topic)
            except GenerationAborted:
                break                         # shutdown mid-card
            except CardRejected as e:
                print(f"[producer] dropped a leaky card: {e}")
                continue                      # rare; just try again, no idle wait
            except Exception as e:
                print(f"[producer] generation failed: {e}")
                self._stop.wait(self.idle_sleep)
                continue
            if self._stop.is_set():
                break
            self.pool.insert_card(topic, text, model=self.model, provisional=False)
            self.shed_idle(topic, text)


# -- standalone pre-fill -------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="pre-fill the welcome-in card pool")
    p.add_argument("--pool-root", default="pool", help="pool directory (default: pool)")
    p.add_argument("--acrostics-csv", default=None,
                   help="acrostics CSV (default: built-in SLOP/TIAT/LEEB)")
    p.add_argument("--producer-model", default="27B", choices=["0.8B", "2B", "9B", "27B"],
                   help="model to generate with (default: 27B)")
    p.add_argument("--pool-cap", type=int, default=100,
                   help="max cards per question (default: 100)")
    p.add_argument("--per-topic", type=int, default=None,
                   help="stop each question at this many cards (for a small "
                        "committed starter set, e.g. 2)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after generating this many cards total")
    p.add_argument("--threads", type=int, default=None, help="llama n_threads")
    p.add_argument("--no-idle", dest="idle", action="store_false",
                   help="don't also shed each card's first stanza into the idle pool")
    args = p.parse_args(argv)

    acrostics = load_acrostics(args.acrostics_csv)
    cap = min(args.pool_cap, args.per_topic) if args.per_topic else args.pool_cap
    pool = CardPool(args.pool_root, cap=cap, signature=acrostics_signature(acrostics))
    # The idle pool is scoped to the first acrostic alone (e.g. pool/SLOP/). Only
    # worth filling when the card has more than one stanza — with a single acrostic
    # the idle pool *is* the main pool, so there's nothing separate to shed.
    idle_pool = (CardPool(args.pool_root, cap=cap,
                          signature=acrostics_signature((acrostics[0],)))
                 if args.idle and len(acrostics) > 1 else None)
    prod = PoolProducer(pool, acrostics, model=args.producer_model,
                        n_threads=args.threads, cap=cap,
                        no_think=(args.producer_model == "27B"), idle_pool=idle_pool)

    # Drive the loop synchronously here (no thread) so --once/--limit are simple and
    # the cards land before the process exits.
    prod._load()
    made = 0
    while True:
        topic = prod._next_topic()
        if topic is None:
            print(f"[producer] pool full ({cap}/topic); done.")
            break
        try:
            text = prod.make_card(topic)
        except CardRejected as e:
            print(f"[producer] dropped a leaky card, retrying topic: {e}")
            continue
        path = pool.insert_card(topic, text, model=args.producer_model)
        prod.shed_idle(topic, text)
        made += 1
        print(f"[producer] {made:4d}  {topic[:50]!r:52}  -> {path}")
        if args.limit is not None and made >= args.limit:
            print(f"[producer] reached --limit {args.limit}; done.")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
