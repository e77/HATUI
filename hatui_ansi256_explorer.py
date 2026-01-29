#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HATUI ANSI-256 Palette Explorer (ANSI-only).

Shows the actual ANSI-256 palette supported by the terminal.
Prints a dense grid with each cell showing the palette index on a colored background.
"""

from __future__ import annotations

import shutil
import sys


ANSI_RESET = "\x1b[0m"


def fg_for_ansi(i: int) -> int:
    if 7 <= i <= 15:
        return 0
    if 187 <= i <= 231:
        return 0
    if 250 <= i <= 255:
        return 0
    if 214 <= i <= 228:
        return 0
    if 33 <= i <= 51:
        return 0
    return 15


def ansi_bg(idx: int) -> str:
    return f"\x1b[48;5;{idx}m"


def ansi_fg(idx: int) -> str:
    return f"\x1b[38;5;{idx}m"


def render_palette() -> str:
    term = shutil.get_terminal_size(fallback=(120, 40))
    width = max(40, term.columns)
    cell_w = 6
    cols = max(1, width // cell_w)
    rows = (256 + cols - 1) // cols

    lines = []
    idx = 0
    for _ in range(rows):
        row = []
        for _ in range(cols):
            if idx >= 256:
                row.append(" " * cell_w)
            else:
                label = f"{idx:03d}".ljust(cell_w)
                fg = fg_for_ansi(idx)
                cell = f"{ansi_bg(idx)}{ansi_fg(fg)}{label}{ANSI_RESET}"
                row.append(cell)
            idx += 1
        lines.append("".join(row))
    return "\n".join(lines)


def main() -> int:
    sys.stdout.write(render_palette())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
