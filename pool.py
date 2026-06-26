"""The card pool: pre-generated question cards on disk, served instantly.

Generation (the slow 27B producer, see producer.py) and serving are decoupled.
The producer writes finished cards here as plain text files; the live kiosk never
runs the model — it pulls the least-viewed pooled card and "types" it to screen.

Storage is a directory of self-describing ``.txt`` files, one card per file, under
``<root>/<signature>/`` (the signature scopes the pool to one acrostic shape, so
changing acrostics.csv serves a fresh set instead of mixing cards). Each file is::

    # topic: <the question string>
    # model: 27B
    # provisional: 0

    <stanza 1 — spells SLOP>

    <stanza 2 — spells TIAT>

    <stanza 3 — spells LEEB>

The header comments carry the metadata (topic self-describing, so it survives a
TOPICS reordering); everything after the first blank line is the card body that
becomes ``Greeting.questions``. View counts are kept **in memory** (reset each
run) — they only need to spread serves across the pool within one session, so
there is no database. A single lock guards the in-memory index and the atomic
file write/delete shared by the producer thread and the serve path.
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


def _slug(signature: str) -> str:
    """Filesystem-safe directory name for an acrostics signature (e.g.
    ``SLOP+TIAT+LEEB`` -> itself; ``SLOP:32-48+...`` -> safe chars only)."""
    return "".join(c if (c.isalnum() or c in "+_-") else "_" for c in signature)


@dataclass(frozen=True)
class PoolCard:
    """A card handed to the serve path: the question it answers, the joined-stanza
    body, which model wrote it, and its (post-increment) view count."""
    topic: str
    text: str
    model: str
    views: int


@dataclass
class _Record:
    """In-memory index entry for one card file."""
    path: Path
    topic: str
    model: str
    provisional: bool
    text: str


def _format_card(topic: str, text: str, model: str, provisional: bool) -> str:
    return (f"# topic: {topic}\n"
            f"# model: {model}\n"
            f"# provisional: {1 if provisional else 0}\n"
            f"\n{text}\n")


def _parse_card(raw: str) -> tuple[dict[str, str], str]:
    """Split a card file into its ``# key: value`` header and the body."""
    header: dict[str, str] = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        key, _, val = lines[i][1:].partition(":")
        header[key.strip()] = val.strip()
        i += 1
    while i < len(lines) and not lines[i].strip():   # skip the blank separator
        i += 1
    return header, "\n".join(lines[i:]).strip()


class CardPool:
    """A directory of pre-generated cards with in-memory view counts.

    Thread-safe across the producer (insert) and the kiosk serve path (take); both
    critical sections are tiny. ``cap`` bounds the number of real cards per topic.
    """

    def __init__(self, root: str | Path = "pool", cap: int = 100,
                 signature: str = "SLOP+TIAT+LEEB"):
        self.signature = signature
        self.cap = cap
        self.dir = Path(root) / _slug(signature)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cards: list[_Record] = []
        self._views: dict[Path, int] = {}
        self._scan()

    # -- loading ---------------------------------------------------------
    def _scan(self) -> None:
        """Read every card file in the signature directory into the index, starting
        each at zero views. Malformed files are skipped, not fatal."""
        for p in sorted(self.dir.glob("*.txt")):
            try:
                header, body = _parse_card(p.read_text())
            except OSError:
                continue
            if not body or "topic" not in header:
                continue
            self._cards.append(_Record(
                path=p, topic=header["topic"], model=header.get("model", "?"),
                provisional=header.get("provisional", "0") == "1", text=body))
            self._views[p] = 0

    # -- serve path ------------------------------------------------------
    def take_least_viewed(self) -> PoolCard | None:
        """Return the globally least-viewed card (random tie-break) and increment
        its in-memory view count, so repeat visitors get fresh cards. ``None`` only
        if the pool is empty (shouldn't happen once warmup seeds one card)."""
        with self._lock:
            if not self._cards:
                return None
            lo = min(self._views[c.path] for c in self._cards)
            pick = random.choice([c for c in self._cards if self._views[c.path] == lo])
            self._views[pick.path] += 1
            return PoolCard(topic=pick.topic, text=pick.text, model=pick.model,
                            views=self._views[pick.path])

    # -- producer / warmup path ------------------------------------------
    def insert_card(self, topic: str, text: str, model: str,
                    provisional: bool = False) -> str | None:
        """Write a card to the pool. The first non-provisional insert evicts the
        0.8B warmup placeholder. Honours the per-topic ``cap`` (returns ``None``
        when that topic is full). The file write is atomic (tmp + os.replace).
        Returns the new file path as a string, or ``None`` if the topic is full."""
        with self._lock:
            if not provisional:
                self._evict_provisional_locked()
            if not provisional and self._topic_count_locked(topic) >= self.cap:
                return None
            path = self.dir / f"{uuid.uuid4().hex[:12]}.txt"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(_format_card(topic, text, model, provisional))
            os.replace(tmp, path)
            self._cards.append(_Record(path=path, topic=topic, model=model,
                                       provisional=provisional, text=text))
            self._views[path] = 0
            return str(path)

    def _evict_provisional_locked(self) -> None:
        keep: list[_Record] = []
        for c in self._cards:
            if c.provisional:
                c.path.unlink(missing_ok=True)
                self._views.pop(c.path, None)
            else:
                keep.append(c)
        self._cards = keep

    def _topic_count_locked(self, topic: str) -> int:
        return sum(1 for c in self._cards if c.topic == topic and not c.provisional)

    # -- introspection (producer balancing, kiosk readout) ---------------
    def counts_by_topic(self) -> dict[str, int]:
        """Real-card count per topic (the lone provisional seed is excluded so it
        doesn't skew the producer's lowest-count-first topic choice)."""
        with self._lock:
            counts: dict[str, int] = {}
            for c in self._cards:
                if not c.provisional:
                    counts[c.topic] = counts.get(c.topic, 0) + 1
            return counts

    def total_count(self) -> int:
        with self._lock:
            return len(self._cards)

    def close(self) -> None:
        """Nothing to flush (view counts are intentionally in-memory); present so
        callers can treat the pool like a resource."""
        return None


# A tiny stamp helper kept here so producer/warmup share one notion of "now" for
# card ordering/debugging without importing time at every call site.
def now_ns() -> int:
    return time.time_ns()
