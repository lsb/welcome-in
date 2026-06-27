"""The card producer: a background worker that pre-fills the pool.

Runs the heavy model (Qwen3.6 27B Q3_K_M, reasoning off) and keeps generating
question cards into the on-disk pool (pool.py) until every question has ``cap``
cards, then idles and tops up after serves/evictions. Serving never waits on it —
the kiosk reads finished cards from the pool and types them out.

When an idle pool is wired in, the producer also fills it as a *separate* pool of
single-acrostic (SLOP) cards for the standing screen's ambient rotation — its own
independent generation runs, balanced against the main pool but never mixed with
it. Both pools share the one loaded model; that's the only thing they have in common.

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
from cards import (CardRejected, GenerationAborted, build_decoders, make_llm,
                   synthesize_card)
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
                 idle_pool: CardPool | None = None,
                 idle_acrostics: tuple[AcrosticSpec, ...] | None = None):
        self.pool = pool
        self.acrostics = acrostics
        self.model = model
        self.topics = topics
        self.n_threads = n_threads
        self.no_think = no_think
        self.cap = cap if cap is not None else pool.cap
        self.idle_sleep = idle_sleep
        # A separate pool of single-acrostic (SLOP) cards for the standing screen's
        # ambient rotation, with its own acrostic set — filled by its own generation
        # runs, never derived from the main cards. Both pools share the one loaded
        # model and nothing else. ``idle_acrostics`` defaults to the first main
        # acrostic alone, which is the shape the kiosk scopes the idle pool to.
        self.idle_pool = idle_pool
        self.idle_acrostics = (idle_acrostics if idle_acrostics is not None
                               else (acrostics[0],) if acrostics else ())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._llm = None
        self._decoders: dict | None = None
        self._idle_decoders: dict | None = None

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
        if self.idle_pool is not None:
            self._idle_decoders = build_decoders(self._llm, self.idle_acrostics,
                                                 no_think=self.no_think,
                                                 temperature=PRODUCER_TEMPERATURE)

    def _targets(self) -> list[tuple[CardPool, tuple[AcrosticSpec, ...], dict]]:
        """The pools this producer fills, each an independent (pool, acrostics,
        decoders) generation target. The idle pool (single-acrostic cards for the
        standing screen) is its own target, generated separately — never derived
        from the main cards. Main is first so ties break toward it."""
        targets = [(self.pool, self.acrostics, self._decoders)]
        if self.idle_pool is not None:
            targets.append((self.idle_pool, self.idle_acrostics, self._idle_decoders))
        return targets

    def _next_job(self):
        """Across every pool this producer fills, the (pool, acrostics, decoders,
        topic) most in need of a card: lowest real-card count, under cap. Ties break
        toward the main pool, then topic order, so the two pools fill in step rather
        than one starving the other. ``None`` when every pool is full to cap."""
        best = None
        for ti, (pool, acrostics, decoders) in enumerate(self._targets()):
            counts = pool.counts_by_topic()
            for tj, topic in enumerate(self.topics):
                n = counts.get(topic, 0)
                if n < self.cap and (best is None or (n, ti, tj) < best[0]):
                    best = ((n, ti, tj), pool, acrostics, decoders, topic)
        return best

    def make_card(self, topic: str, acrostics: tuple[AcrosticSpec, ...] | None = None,
                  decoders: dict | None = None) -> str:
        """Generate one card for ``topic`` from the given acrostics/decoders (the
        main set by default; the idle set when filling the idle pool)."""
        return synthesize_card(decoders or self._decoders,
                               acrostics or self.acrostics, topic,
                               seed_base=time.time_ns() & 0x7FFFFFFF,
                               abort=self._stop.is_set)

    def _run(self) -> None:
        try:
            self._load()
        except Exception as e:  # never take down the kiosk over a producer failure
            print(f"[producer] load failed: {e}")
            return
        sigs = " + ".join(p.signature for p, _, _ in self._targets())
        print(f"[producer] {self.model} ready; filling {sigs} to {self.cap}/topic")
        while not self._stop.is_set():
            job = self._next_job()
            if job is None:                   # all pools full -> idle, top up later
                self._stop.wait(self.idle_sleep)
                continue
            _, pool, acrostics, decoders, topic = job
            try:
                text = self.make_card(topic, acrostics, decoders)
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
            pool.insert_card(topic, text, model=self.model, provisional=False)


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
                   help="don't fill the separate idle pool of single-acrostic cards")
    args = p.parse_args(argv)

    acrostics = load_acrostics(args.acrostics_csv)
    cap = min(args.pool_cap, args.per_topic) if args.per_topic else args.pool_cap
    pool = CardPool(args.pool_root, cap=cap, signature=acrostics_signature(acrostics))
    # The idle pool is its own pool of single-acrostic cards (the first acrostic
    # alone, e.g. pool/SLOP/), filled independently. Only worth a separate pool when
    # the card has more than one stanza — with a single acrostic it would just be the
    # main pool over again.
    idle_acrostics = (acrostics[0],) if acrostics else ()
    idle_pool = (CardPool(args.pool_root, cap=cap,
                          signature=acrostics_signature(idle_acrostics))
                 if args.idle and len(acrostics) > 1 else None)
    prod = PoolProducer(pool, acrostics, model=args.producer_model,
                        n_threads=args.threads, cap=cap,
                        no_think=(args.producer_model == "27B"),
                        idle_pool=idle_pool, idle_acrostics=idle_acrostics)

    # Drive the loop synchronously here (no thread) so --once/--limit are simple and
    # the cards land before the process exits.
    prod._load()
    made = 0
    while True:
        job = prod._next_job()
        if job is None:
            print(f"[producer] pools full ({cap}/topic); done.")
            break
        _, jpool, jacr, jdec, topic = job
        try:
            text = prod.make_card(topic, jacr, jdec)
        except CardRejected as e:
            print(f"[producer] dropped a leaky card, retrying topic: {e}")
            continue
        path = jpool.insert_card(topic, text, model=args.producer_model)
        made += 1
        print(f"[producer] {made:4d}  [{jpool.signature}]  {topic[:40]!r:42}  -> {path}")
        if args.limit is not None and made >= args.limit:
            print(f"[producer] reached --limit {args.limit}; done.")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
