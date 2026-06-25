"""Print the finished question card to a CUPS printer via ``lp``.

The visitor leaves with a physical card. Generation streams to the screen live
(greeter.py / acrostic.py), but the print is a single shot once the card is
complete and its last line settled. ``lp`` reads the card on stdin and the
system spooler (CUPS — on the Pi and on macOS alike) does the rest.

We render the card as a one-page **PostScript** document (``lp`` autodetects it
from the ``%!PS`` magic) so it can carry real typography:

  * Part 1 — the spoken hello — is set in Times-Italic, a serif *proportional*
    italic face, wrapped at 80 columns.
  * the theme follows, centered and italic.
  * Part 2 — the acrostic question card — is set in fixed-width Courier so the
    lines column up, and the first character of every line is **bold** so the
    hidden first-letter acrostic reads at a glance down the left edge.

The page is rotated into **landscape** in the prologue (as the old ``lp -o
landscape`` did) so the ~60-100 char acrostic lines never wrap. A specific queue
can be named; otherwise CUPS's default printer is used. The whole thing is
optional — with no printer attached (``WELCOME_PRINTER=off``) the sink no-ops,
so the kiosk still runs and streams to the screen.

Env:
  WELCOME_PRINTER   queue name to print to; "off"/"none"/"no" disables printing;
                    unset -> the system default printer.
  WELCOME_PRINT_CPI characters-per-inch for the fixed-width body (default 12);
                    Courier point size = 120 / cpi, so 12 cpi -> 10 pt.
"""

from __future__ import annotations

import os
import subprocess
import textwrap

_OFF = frozenset({"off", "none", "no"})


# Typographic Unicode -> ASCII so the printed card can't mojibake regardless of
# the Pi's locale or which CUPS filter renders it: a dist-upgrade can reset the
# locale to non-UTF-8. Keeping the PostScript pure ASCII also means we only ever
# emit code points the base-14 fonts encode, with no font-encoding surprises.
# The acrostic body is already ASCII by grammar; this also flattens the
# "— topic —" wrapper and any stray char the raw hello slips in.
_ASCII_MAP = {
    "—": "-", "–": "-", "―": "-", "…": "...", "·": "*", "•": "*",
    "→": "->", "↔": "<->", "±": "+/-",
    "“": '"', "”": '"', "‘": "'", "’": "'", " ": " ",
}


def _asciify(text: str) -> str:
    for u, a in _ASCII_MAP.items():
        text = text.replace(u, a)
    # Drop anything still non-ASCII so the PostScript is pure ASCII and no
    # filter or font encoding can misrender it.
    return text.encode("ascii", "ignore").decode("ascii")


def _ps_escape(text: str) -> str:
    """Escape a string for a PostScript literal ``(...)``: backslash first, then
    the parens that delimit the literal."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# Letter media is portrait (612 x 792 pt); the prologue rotates it so we draw in
# a 792 x 612 logical space, origin lower-left, working top-down.
_PAGE_W, _PAGE_H = 792.0, 612.0
_MARGIN = 40.0
_HELLO_PT, _HELLO_LEAD = 15.0, 19.0   # Part 1: serif proportional italic
_HELLO_WRAP = 80                      # columns to wrap the hello at
_TOPIC_PT, _TOPIC_LEAD = 13.0, 26.0   # the theme, centered italic
_GAP_AFTER_HELLO = 24.0               # set the hello apart from the card below
_BYLINE = "by Lee Butterman"          # signature pinned to the lower-left corner
_BYLINE_PT = 10.0                     # small serif italic, like the screen byline


def card_postscript(greeting, cpi: int = 12) -> str:
    """Render the finished card as a one-page PostScript document for ``lp``.

    The hello is wrapped at 80 columns and set in a serif proportional italic;
    the theme is centered and italic; the question card is fixed-width with the
    first character of each line bolded to surface the acrostic. The body point
    size is derived from ``cpi`` (Courier: point size = 120 / cpi) so the
    WELCOME_PRINT_CPI knob still tunes the body density."""
    body_pt = 120.0 / max(1, cpi)
    body_lead = body_pt * 1.3

    out = [
        "%!PS-Adobe-3.0",
        "%%Creator: welcome-in",
        "%%Orientation: Landscape",
        "%%Pages: 1",
        "%%EndComments",
        "%%Page: 1 1",
        # Rotate portrait Letter into landscape: logical (x,y) -> device
        # (612 - y, x), giving a 792-wide x 612-tall canvas so the wide acrostic
        # lines never wrap, just as the old `lp -o landscape` ensured.
        "612 0 translate 90 rotate",
    ]

    y = _PAGE_H - 50.0

    # Part 1 — the hello: serif proportional italic, wrapped at 80 columns.
    hello = _asciify(greeting.hello).strip()
    if hello:
        out.append(f"/Times-Italic {_HELLO_PT:g} selectfont")
        for line in textwrap.wrap(hello, width=_HELLO_WRAP):
            out.append(f"{_MARGIN:g} {y:.1f} moveto ({_ps_escape(line)}) show")
            y -= _HELLO_LEAD
    y -= _GAP_AFTER_HELLO

    # The theme — centered. `stringwidth` measures the proportional font for us,
    # so we don't have to know the glyph metrics to center it.
    topic = _asciify(f"- {greeting.topic} -").strip()
    if topic:
        out.append(f"/Times-Italic {_TOPIC_PT:g} selectfont")
        out.append(f"({_ps_escape(topic)}) dup stringwidth pop 2 div "
                   f"{_PAGE_W / 2:g} exch sub {y:.1f} moveto show")
        y -= _TOPIC_LEAD

    # Part 2 — the question card: fixed-width Courier, first char of each line
    # bold to surface the hidden acrostic down the left edge.
    for line in _asciify(greeting.questions).split("\n"):
        if line:
            first, rest = line[0], line[1:]
            out.append(
                f"{_MARGIN:g} {y:.1f} moveto "
                f"/Courier-Bold {body_pt:g} selectfont ({_ps_escape(first)}) show "
                f"/Courier {body_pt:g} selectfont ({_ps_escape(rest)}) show")
        y -= body_lead

    # Signature pinned to the lower-left corner, at a fixed baseline above the
    # bottom margin so it sits in the same spot regardless of how the card flows.
    out.append(f"/Times-Italic {_BYLINE_PT:g} selectfont")
    out.append(f"{_MARGIN:g} {_MARGIN:.1f} moveto ({_ps_escape(_asciify(_BYLINE))}) show")

    out.append("showpage")
    return "\n".join(out) + "\n"


class Printer:
    def __init__(self, queue: str | None = None, enabled: bool = True, cpi: int = 12):
        self.queue = queue
        self.enabled = enabled
        self.cpi = cpi

    @classmethod
    def from_env(cls) -> "Printer":
        q = os.environ.get("WELCOME_PRINTER", "").strip()
        if q.lower() in _OFF:
            return cls(enabled=False)
        cpi = int(os.environ.get("WELCOME_PRINT_CPI", "12"))
        return cls(queue=(q or None), enabled=True, cpi=cpi)

    def print_card(self, greeting) -> None:
        """Print the finished card. Quietly reports (rather than raises) on
        failure, so a printer hiccup never takes down the greeter. ``lp``
        autodetects the PostScript from its `%!PS` header and lays out the page
        itself (the document carries its own landscape orientation and fonts),
        so no `-o landscape`/`cpi` text options are needed."""
        if not self.enabled:
            return
        cmd = ["lp", "-t", "welcome-in"]
        if self.queue:
            cmd[1:1] = ["-d", self.queue]
        try:
            subprocess.run(cmd, input=card_postscript(greeting, self.cpi).encode("ascii"),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as e:
            detail = e.stderr.decode().strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else e
            print(f"[printer] could not print the card: {detail}")
