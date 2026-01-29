#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HATUI ANSI ruler / layout helper.

Prints a simple horizontal and vertical ruler grid to help debug terminal layouts.
"""

from __future__ import annotations

import shutil
import sys


def build_ruler(width: int) -> str:
    digits = "".join(str((i // 10) % 10) for i in range(width))
    ones = "".join(str(i % 10) for i in range(width))
    return f"{digits}\n{ones}"


def build_grid(width: int, height: int) -> str:
    lines = []
    header = build_ruler(width)
    lines.extend(header.splitlines())
    for row in range(height - 2):
        prefix = f"{row:03d}"
        line = prefix + " " + ("." * max(0, width - len(prefix) - 1))
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    term = shutil.get_terminal_size(fallback=(120, 40))
    width = max(20, term.columns)
    height = max(5, term.lines)
    sys.stdout.write(build_grid(width, height))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
