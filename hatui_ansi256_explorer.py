#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HATUI ANSI-256 Palette Explorer (Textual 0.76 / tty1 friendly)

Shows the *actual* colours your tty1 reliably supports: ANSI 256 (color(0)..color(255))

Layout:
- Auto-fits a dense grid across the full screen.
- Each cell shows the palette index (000-255) on a coloured background.

Controls:
- Arrow keys / hjkl : move selection
- Enter            : pick selected colour -> saved as last_pick (e.g. color(214))
- g                : toggle grid label mode (numbers only vs numbers+coords)
- p                : toggle page mode (all vs 0-127 vs 128-255) (bigger cells)
- q / Esc          : quit

Tip:
- Use picked values directly in your YAML:  color_normal: "color(214)"
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Static
from textual import events
from rich.text import Text


def coord_col_name(n: int) -> str:
    # 0 -> A, 25 -> Z, 26 -> AA ...
    s = ""
    n = int(n)
    while True:
        n, r = divmod(n, 26)
        s = chr(ord("A") + r) + s
        if n == 0:
            break
        n -= 1
    return s


def fg_for_ansi(i: int) -> str:
    """Reasonable foreground for readability without needing RGB tables."""
    if 7 <= i <= 15:
        return "black"
    if 187 <= i <= 231:
        return "black"
    if 250 <= i <= 255:
        return "black"
    if 214 <= i <= 228:
        return "black"
    if 33 <= i <= 51:
        return "black"
    return "white"


class AnsiGrid(Static):
    def render(self) -> Text:
        app = self.app  # type: ignore[attr-defined]
        w = self.size.width
        h = self.size.height

        if app.page_mode:
            cell_w = 9
            cell_h = 2
        else:
            cell_w = 6
            cell_h = 1

        cols = max(1, w // cell_w)
        rows = max(1, h // cell_h)

        start, end = app.palette_range()
        total = end - start + 1
        max_cells = cols * rows
        show_n = min(total, max_cells)

        t = Text()
        for r in range(rows):
            for line in range(cell_h):
                for c in range(cols):
                    cell_idx = r * cols + c
                    pal_idx = start + cell_idx
                    if cell_idx >= show_n:
                        t.append(" " * cell_w)
                        continue

                    bg = f"color({pal_idx})"
                    fg = fg_for_ansi(pal_idx)

                    if app.show_coords:
                        coord = f"{coord_col_name(c)}{r+1}"
                        label = f"{pal_idx:03d}"
                        if cell_h == 2:
                            content = (label if line == 0 else coord)[:cell_w].ljust(cell_w)
                        else:
                            content = label[:cell_w].ljust(cell_w)
                    else:
                        content = f"{pal_idx:03d}".ljust(cell_w)

                    content = content[:cell_w].ljust(cell_w)

                    style = f"{fg} on {bg}"
                    if r == app.sel_r and c == app.sel_c:
                        t.append(content, style=style + " bold underline")
                    else:
                        t.append(content, style=style)

                if not (r == rows - 1 and line == cell_h - 1):
                    t.append("\n")

        return t


class HatuiAnsi256Explorer(App):
    CSS = """\
Screen { background: black; }
#title  { height: 3; padding: 0 1; color: #ffb000; }
#grid   { height: 1fr; width: 100%; }
#status { height: 3; padding: 0 1; color: #ffb000; }
"""

    sel_r: reactive[int] = reactive(0)
    sel_c: reactive[int] = reactive(0)
    last_pick: reactive[str] = reactive("")
    show_coords: reactive[bool] = reactive(False)
    page_mode: reactive[int] = reactive(0)  # 0=all, 1=0-127, 2=128-255

    def compose(self) -> ComposeResult:
        yield Static(id="title")
        yield AnsiGrid(id="grid")
        yield Static(id="status")

    def on_mount(self) -> None:
        self._title = self.query_one("#title", Static)
        self._grid = self.query_one("#grid", AnsiGrid)
        self._status = self.query_one("#status", Static)
        self.set_interval(0.5, self._refresh)
        self._refresh()

    def palette_range(self):
        if self.page_mode == 1:
            return 0, 127
        if self.page_mode == 2:
            return 128, 255
        return 0, 255

    def _dims(self):
        w = self._grid.size.width
        h = self._grid.size.height
        if self.page_mode:
            cell_w = 9
            cell_h = 2
        else:
            cell_w = 6
            cell_h = 1
        cols = max(1, w // cell_w)
        rows = max(1, h // cell_h)
        return cols, rows, cell_w, cell_h

    def selected_index(self) -> int:
        cols, rows, *_ = self._dims()
        start, end = self.palette_range()
        idx = start + self.sel_r * cols + self.sel_c
        if idx < start:
            return start
        if idx > end:
            return end
        return idx

    def _refresh(self) -> None:
        cols, rows, cell_w, cell_h = self._dims()
        mode = {0: "ALL(0-255)", 1: "PAGE 0-127", 2: "PAGE 128-255"}[self.page_mode]
        coord_mode = "COORDS" if self.show_coords else "NUMBERS"
        self._title.update(
            Text(
                f"HATUI ANSI-256 PALETTE  |  {mode}  |  {coord_mode}\n"
                f"Grid: {cols}x{rows}  Cell:{cell_w}x{cell_h}  Keys: arrows/hjkl, Enter pick, g coords, p page, q quit"
            )
        )

        sel = self.selected_index()
        self._status.update(
            Text(
                f"Selected: color({sel})   Index: {sel:03d}   Last pick: {self.last_pick}\n"
                f"Use in YAML:  color_normal: \"color({sel})\"   (press p for bigger cells)"
            )
        )
        self._grid.refresh()

    async def on_key(self, event: events.Key) -> None:
        key = event.key
        cols, rows, *_ = self._dims()

        if key in ("q", "escape"):
            raise SystemExit(0)

        if key in ("left", "h"):
            self.sel_c = (self.sel_c - 1) % cols
        elif key in ("right", "l"):
            self.sel_c = (self.sel_c + 1) % cols
        elif key in ("up", "k"):
            self.sel_r = max(0, self.sel_r - 1)
        elif key in ("down", "j"):
            self.sel_r = min(rows - 1, self.sel_r + 1)

        elif key == "g":
            self.show_coords = not self.show_coords

        elif key == "p":
            self.page_mode = (self.page_mode + 1) % 3
            self.sel_r = 0
            self.sel_c = 0

        elif key == "enter":
            sel = self.selected_index()
            self.last_pick = f"color({sel})"

        self._refresh()


if __name__ == "__main__":
    HatuiAnsi256Explorer().run()
