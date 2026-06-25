"""Print the finished question card to a CUPS printer via ``lp``.

The visitor leaves with a physical card. Generation streams to the screen live
(greeter.py / acrostic.py), but the print is a single shot once the card is
complete and its last line settled. ``lp`` reads the card text on stdin and the
system spooler (CUPS — on the Pi and on macOS alike) does the rest.

Plain text is all we need: the acrostic lines run ~60-100 chars, so we print
**landscape** at 12 cpi (~120 columns on Letter) so no line wraps. A specific
queue can be named; otherwise CUPS's default printer is used. The whole thing is
optional — with no printer attached (``WELCOME_PRINTER=off``) the sink no-ops, so
the kiosk still runs and streams to the screen.

Env:
  WELCOME_PRINTER   queue name to print to; "off"/"none"/"no" disables printing;
                    unset -> the system default printer.
  WELCOME_PRINT_CPI characters-per-inch for the landscape text (default 12).
"""

from __future__ import annotations

import os
import subprocess

_OFF = frozenset({"off", "none", "no"})


def format_card(greeting) -> str:
    """The plain-text card: the spoken hello, the theme, then the questions.

    Three newlines (two blank lines) set the short spoken hello apart from the
    theme + question card that follows; single blank lines separate the rest."""
    return "\n".join([
        greeting.hello,
        "", "",
        f"— {greeting.topic} —",
        "",
        greeting.questions,
        "",
    ])


# Typographic Unicode -> ASCII so the printed card can't mojibake regardless of
# the Pi's locale or which CUPS text filter renders it: a dist-upgrade can reset
# the locale to non-UTF-8, and a legacy text filter then prints UTF-8 bytes as
# Latin-1 ("—" comes out as "â€""). The acrostic body is already ASCII by grammar;
# this also flattens the "— topic —" wrapper and any stray char the raw hello slips in.
_ASCII_MAP = {
    "—": "-", "–": "-", "―": "-", "…": "...", "·": "*", "•": "*",
    "→": "->", "↔": "<->", "±": "+/-",
    "“": '"', "”": '"', "‘": "'", "’": "'", " ": " ",
}


def _asciify(text: str) -> str:
    for u, a in _ASCII_MAP.items():
        text = text.replace(u, a)
    # Drop anything still non-ASCII so lp's stdin is pure ASCII and no text
    # filter can misrender it.
    return text.encode("ascii", "ignore").decode("ascii")


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
        failure, so a printer hiccup never takes down the greeter."""
        if not self.enabled:
            return
        cmd = ["lp", "-o", "landscape", "-o", f"cpi={self.cpi}", "-o", "lpi=6",
               "-t", "welcome-in"]
        if self.queue:
            cmd[1:1] = ["-d", self.queue]
        try:
            subprocess.run(cmd, input=_asciify(format_card(greeting)).encode("ascii"), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as e:
            detail = e.stderr.decode().strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else e
            print(f"[printer] could not print the card: {detail}")
