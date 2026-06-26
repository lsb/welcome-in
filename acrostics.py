"""The set of acrostics that make up one question card.

A card is built from several short acrostic *stanzas* — three separate decode
runs, one per secret in this list (default SLOP / TIAT / LEEB), concatenated.
Separate short runs stay crisp where one long acrostic pads and degenerates (see
the model-ladder notes), so the card's three stanzas are generated independently
and the whole card's left edge spells the secrets in order: SLOP TIAT LEEB.

The list is configurable from a CSV (``acrostics.csv``) so the hidden words — and
optionally their per-stanza line bounds — can change without code edits. Each row
is ``secret[,min_line[,max_line]]``; a header row (first cell ``secret``) is
optional. ``secret`` is cleaned to ASCII letters, mirroring
``acrostic.acrostic_from_env``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AcrosticSpec:
    """One stanza's acrostic: the secret it spells, plus optional per-stanza line
    bounds (``None`` falls back to ``acrostic.DEFAULT_MIN_LINE`` / ``MAX_LINE``)."""
    secret: str
    min_line: int | None = None
    max_line: int | None = None


# The shipped triple. One card = a SLOP stanza, a TIAT stanza, and a LEEB stanza.
DEFAULT_ACROSTICS: tuple[AcrosticSpec, ...] = (
    AcrosticSpec("SLOP"),
    AcrosticSpec("TIAT"),
    AcrosticSpec("LEEB"),
)


def _clean_secret(raw: str) -> str:
    """ASCII letters only (drops spaces, digits, punctuation) — same rule the
    environment override uses, so a CSV and ``WELCOME_ACROSTIC`` behave alike."""
    return "".join(c for c in raw if c.isascii() and c.isalpha())


def _to_int(cell: str) -> int | None:
    cell = cell.strip()
    return int(cell) if cell else None


def load_acrostics(path: str | Path | None = None) -> tuple[AcrosticSpec, ...]:
    """Load the card's stanza acrostics from a CSV, or the shipped default triple.

    Rows are ``secret[,min_line[,max_line]]``. A leading header row whose first
    cell is ``secret`` (any case) or non-alphabetic is skipped. Rows whose secret
    cleans to empty are dropped. A missing/None path, a missing file, or an empty
    file all fall back to ``DEFAULT_ACROSTICS``.
    """
    if path is None:
        return DEFAULT_ACROSTICS
    p = Path(path)
    if not p.is_file():
        return DEFAULT_ACROSTICS

    specs: list[AcrosticSpec] = []
    with p.open(newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row or not row[0].strip():
                continue
            first = row[0].strip()
            if i == 0 and (first.lower() == "secret" or not first.isalpha()):
                continue  # header row
            secret = _clean_secret(first)
            if not secret:
                continue
            min_line = _to_int(row[1]) if len(row) > 1 else None
            max_line = _to_int(row[2]) if len(row) > 2 else None
            specs.append(AcrosticSpec(secret, min_line, max_line))
    return tuple(specs) or DEFAULT_ACROSTICS


def acrostics_signature(specs: tuple[AcrosticSpec, ...]) -> str:
    """A stable identifier for a card's acrostic set, e.g. ``SLOP+TIAT+LEEB`` (or
    ``SLOP:32-48+...`` when a row overrides its line bounds). Stored on every
    pooled card and used to scope the pool directory, so changing the CSV serves a
    fresh set instead of mixing cards built to different shapes."""
    parts = []
    for s in specs:
        if s.min_line is None and s.max_line is None:
            parts.append(s.secret)
        else:
            parts.append(f"{s.secret}:{s.min_line or ''}-{s.max_line or ''}")
    return "+".join(parts)
