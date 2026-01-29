#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HATUI Ruler / Layout Debugger (Textual)

Purpose:
- Show actual widget/panel measurements in characters (width/height).
- Provide visual "rulers" so we can see where Textual starts cropping/ellipsizing.
- Keep the 3-column grid layout (Left / Middle / Right) with header + footer.

Compatible with Textual 0.76.x
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static
from textual import on
from rich.text import Text


def clamp(n: int, lo: int, hi: int) -> int:
    return lo if n < lo else hi if n > hi else n


def make_ruler(width: int, every: int = 10) -> str:
    """Return a fixed-width ruler string 0..width-1 with tick labels every N chars."""
    if width <= 0:
        return ""
    s = [" "] * width
    for i in range(width):
        if i % every == 0:
            s[i] = "|"
        elif i % 5 == 0:
            s[i] = "+"
        else:
            s[i] = "-"
    # place numeric labels
    label_positions = list(range(0, width, every))
    for pos in label_positions:
        lab = str(pos)
        for j, ch in enumerate(lab):
            if pos + j < width:
                s[pos + j] = ch
    return "".join(s)


def make_span_line(width: int) -> str:
    """A line that is exactly width chars, to test truncation. Ends with 'END' marker."""
    if width <= 0:
        return ""
    core = ["·"] * width
    end = "END"
    start = max(0, width - len(end))
    for i, ch in enumerate(end):
        if start + i < width:
            core[start + i] = ch
    # also put START at beginning if fits
    start_lab = "START"
    for i, ch in enumerate(start_lab):
        if i < width:
            core[i] = ch
    return "".join(core)


class Box(Static):
    """A bordered box that renders measurement + ruler content."""

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.title = title

    def render(self) -> Text:
        w = self.size.width
        h = self.size.height
        # inner guess: account for 2-char border (left/right) when using border in CSS
        inner_w_guess = max(0, w - 2)

        now = datetime.now().strftime("%H:%M:%S")
        t = Text()
        t.append(f"{self.title}\n", style="bold")
        t.append(f"time: {now}\n")
        t.append(f"size: {w}w x {h}h chars\n")
        t.append(f"inner guess (w-2): {inner_w_guess}\n\n")

        # Ruler based on full widget width and inner guess
        t.append("RULER (full width)\n", style="bold")
        t.append(Text(make_ruler(clamp(w, 0, 300), every=10)[:w], no_wrap=True, overflow="crop"))
        t.append("\n")
        t.append("SPAN TEST (full width)\n", style="bold")
        t.append(Text(make_span_line(clamp(w, 0, 300))[:w], no_wrap=True, overflow="crop"))
        t.append("\n\n")

        t.append("RULER (inner guess)\n", style="bold")
        t.append(Text(make_ruler(clamp(inner_w_guess, 0, 300), every=10)[:inner_w_guess], no_wrap=True, overflow="crop"))
        t.append("\n")
        t.append("SPAN TEST (inner guess)\n", style="bold")
        t.append(Text(make_span_line(clamp(inner_w_guess, 0, 300))[:inner_w_guess], no_wrap=True, overflow="crop"))
        t.append("\n")

        return t


class HeaderBar(Static):
    def render(self) -> Text:
        w = self.size.width
        h = self.size.height
        env = os.getenv("TERM", "?")
        mode = os.getenv("HATUI_MODE", "RULER")
        txt = Text()
        txt.append(f"MODE:{mode}   ")
        txt.append(f"TERM:{env}   ")
        txt.append(f"HEADER:{w}x{h}   ")
        txt.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return txt


class FooterBar(Static):
    heartbeat = reactive(0)

    def on_mount(self) -> None:
        self.set_interval(0.25, self._tick)

    def _tick(self) -> None:
        self.heartbeat += 1

    def render(self) -> Text:
        w = self.size.width
        h = self.size.height
        frames = "|/-\\"
        sp = frames[self.heartbeat % len(frames)]
        txt = Text()
        txt.append(f"FOOTER:{w}x{h}   ")
        txt.append(f"HB:{sp}   ")
        txt.append("Tip: resize font/console and see widths change.")
        return txt


class RulerApp(App):
    CSS = """
    Screen {
        background: black;
        color: #ffb000;  /* your normal */
    }
    #root {
        height: 100%;
        width: 100%;
    }
    #header {
        height: 3;
        padding: 0 1;
        border: round #ffb000;
    }
    #footer {
        height: 3;
        padding: 0 1;
        border: round #ffb000;
    }
    #main {
        height: 1fr;
        width: 100%;
    }
    .panel {
        height: 100%;
        width: 1fr;
        padding: 0 1;
        border: round #ffb000;
    }
    """


    def compose(self) -> ComposeResult:
        yield Container(
            HeaderBar(id="header"),
            Horizontal(
                Box("LEFT PANEL", id="left", classes="panel"),
                Box("MIDDLE PANEL", id="mid", classes="panel"),
                Box("RIGHT PANEL", id="right", classes="panel"),
                id="main",
            ),
            FooterBar(id="footer"),
            id="root",
        )

    def on_mount(self) -> None:
        # refresh all panels periodically so you can see time + live sizes
        self.set_interval(0.5, self._refresh)

    def _refresh(self) -> None:
        for wid in ("left", "mid", "right", "header", "footer"):
            try:
                self.query_one(f"#{wid}").refresh()
            except Exception:
                pass


if __name__ == "__main__":
    RulerApp().run()
