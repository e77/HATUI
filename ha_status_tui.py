#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import yaml

# Do not introduce Textual or other UI frameworks.

__version__ = "9.6.6-fix2"
__build__ = "2026-01-28"

SPINNER_FRAMES = ["|", "/", "-", "\\"]

LOG_PATH = os.getenv("HATUI_LOG", "/tmp/hatui.log")
CTL_PATH = os.getenv("HATUI_CTL", "/run/hatui/ctl")
FIXTURE_PATH = os.getenv("HATUI_FIXTURE", "fixtures.yaml").strip()

ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_HIDE_CURSOR = "\x1b[?25l"
ANSI_SHOW_CURSOR = "\x1b[?25h"
ANSI_CLEAR = "\x1b[2J\x1b[H"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log(f"YAML read error for {path}: {e}")
        return {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def log(msg: str) -> None:
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


@dataclass
class EntityState:
    state: str
    attributes: Dict[str, Any]
    last_changed: str


class HAClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def get_state(self, session: aiohttp.ClientSession, entity_id: str) -> EntityState:
        url = f"{self.base_url}/api/states/{entity_id}"
        async with session.get(url, headers=self._headers(), timeout=self.timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HA /states {resp.status}: {text[:200]}")
            data = await resp.json()
            return EntityState(
                state=str(data.get("state", "")),
                attributes=dict(data.get("attributes", {}) or {}),
                last_changed=str(data.get("last_changed", "")),
            )

    async def get_history(
        self,
        session: aiohttp.ClientSession,
        entity_id: str,
        start_utc: datetime,
        end_utc: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        start_iso = start_utc.astimezone(timezone.utc).isoformat()
        url = f"{self.base_url}/api/history/period/{start_iso}"
        params: Dict[str, str] = {
            "filter_entity_id": entity_id,
            "minimal_response": "1",
        }
        if end_utc is not None:
            params["end_time"] = end_utc.astimezone(timezone.utc).isoformat()

        async with session.get(url, headers=self._headers(), params=params, timeout=self.timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HA /history {resp.status}: {text[:200]}")
            data = await resp.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                return data[0]
            if isinstance(data, list):
                return data
            return []


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def build_bool_bar(
    history: List[Dict[str, Any]],
    now_utc: datetime,
    window_hours: float,
    blocks: int,
    predicate,
) -> List[bool]:
    start = now_utc - timedelta(hours=window_hours)
    events: List[Tuple[datetime, bool]] = []

    for row in history or []:
        ts = parse_iso(str(row.get("last_changed") or row.get("last_updated") or ""))
        if ts is None:
            continue
        events.append((ts, bool(predicate(row.get("state")))))

    events.sort(key=lambda t: t[0])

    current = False
    for ts, is_on in events:
        if ts <= start:
            current = is_on
        else:
            break

    out: List[bool] = []
    block_seconds = (window_hours * 3600) / max(1, int(blocks))
    eidx = 0
    window_start = start

    for b in range(int(blocks)):
        window_end = start + timedelta(seconds=(b + 1) * block_seconds)
        while eidx < len(events) and events[eidx][0] < window_end:
            if events[eidx][0] >= window_start:
                current = events[eidx][1]
            eidx += 1
        out.append(current)
        window_start = window_end

    return out


def build_state_bar_24h(
    history: List[Dict[str, Any]],
    now_utc: datetime,
    on_states: List[str],
    blocks: int = 96,
) -> List[bool]:
    # Return bool list across last 24h; blocks=96 => 15-min resolution.
    start = now_utc - timedelta(hours=24)
    events: List[Tuple[datetime, bool]] = []

    on_set = set([s.lower() for s in on_states])
    for row in history or []:
        ts = parse_iso(str(row.get("last_changed") or row.get("last_updated") or ""))
        if ts is None:
            continue
        st = str(row.get("state", "")).strip().lower()
        events.append((ts, st in on_set))

    events.sort(key=lambda t: t[0])

    current = False
    for ts, is_on in events:
        if ts <= start:
            current = is_on
        else:
            break

    out: List[bool] = []
    block_seconds = (24 * 3600) / max(1, int(blocks))
    eidx = 0
    window_start = start

    for b in range(int(blocks)):
        window_end = start + timedelta(seconds=(b + 1) * block_seconds)
        while eidx < len(events) and events[eidx][0] < window_end:
            if events[eidx][0] >= window_start:
                current = events[eidx][1]
            eidx += 1
        out.append(current)
        window_start = window_end

    return out


def parse_active_when(expr: str) -> Tuple[str, Optional[str], Optional[float]]:
    expr = str(expr or "").strip()
    if not expr:
        return "", None, None
    m = re.match(r"^(\S+)\s*(>=|<=|==|!=|>|<|=)\s*([0-9.]+)$", expr)
    if m:
        entity_id, op, value = m.groups()
        return entity_id.strip(), ("==" if op == "=" else op), safe_float(value)
    return expr, None, None


def compute_period_fractions(
    history: List[Dict[str, Any]],
    now_utc: datetime,
    on_states: List[str],
    window_hours: float,
    periods: int,
) -> Tuple[List[float], Optional[datetime], bool]:
    """Return (fractions, last_on_end_utc, current_on).

    fractions: length 'periods', each in [0..1] representing on-time fraction in that period.
    last_on_end_utc: last instant the boiler was ON within the window (or None if never).
    current_on: whether the boiler is ON at now_utc (based on latest state at end).
    """
    window_hours = float(window_hours)
    periods = int(periods)
    start = now_utc - timedelta(hours=window_hours)
    end = now_utc

    on_set = set([s.lower() for s in on_states])

    # Extract timestamped events
    events: List[Tuple[datetime, bool]] = []
    for row in history or []:
        ts = parse_iso(str(row.get("last_changed") or row.get("last_updated") or ""))
        if ts is None:
            continue
        st = str(row.get("state", "")).strip().lower()
        events.append((ts, st in on_set))
    events.sort(key=lambda t: t[0])

    # Determine state at window start
    current = False
    for ts, is_on in events:
        if ts <= start:
            current = is_on
        else:
            break

    # Build segments within [start, end]
    segs: List[Tuple[datetime, datetime, bool]] = []
    last_ts = start
    idx = 0
    while idx < len(events) and events[idx][0] <= start:
        idx += 1

    while idx < len(events):
        ts, is_on = events[idx]
        if ts >= end:
            break
        if ts > last_ts:
            segs.append((last_ts, ts, current))
        current = is_on
        last_ts = ts
        idx += 1

    if last_ts < end:
        segs.append((last_ts, end, current))

    total_sec = (end - start).total_seconds()
    period_sec = total_sec / float(periods) if periods else total_sec
    on_seconds = [0.0 for _ in range(periods)]
    last_on_end: Optional[datetime] = None

    for a, b, is_on in segs:
        if is_on:
            last_on_end = b
        if not is_on:
            continue
        seg_start = (a - start).total_seconds()
        seg_end = (b - start).total_seconds()
        p0 = int(max(0, min(periods - 1, seg_start // period_sec)))
        p1 = int(max(0, min(periods - 1, (seg_end - 1e-6) // period_sec)))
        for p in range(p0, p1 + 1):
            ps = p * period_sec
            pe = (p + 1) * period_sec
            ov = max(0.0, min(seg_end, pe) - max(seg_start, ps))
            if ov > 0:
                on_seconds[p] += ov

    fractions: List[float] = []
    for p in range(periods):
        frac = on_seconds[p] / period_sec if period_sec > 0 else 0.0
        frac = 0.0 if frac < 0.0 else 1.0 if frac > 1.0 else frac
        fractions.append(frac)

    current_on = bool(current)
    return fractions, last_on_end, current_on


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def crop_ansi(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if visible_len(text) <= width:
        return text
    out = []
    count = 0
    i = 0
    while i < len(text) and count < width:
        if text[i] == "\x1b":
            m = ANSI_RE.match(text, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        out.append(text[i])
        count += 1
        i += 1
    out.append(ANSI_RESET)
    return "".join(out)


def pad_ansi(text: str, width: int, align: str = "left") -> str:
    text = crop_ansi(text, width)
    length = visible_len(text)
    if length >= width:
        return text
    pad = width - length
    if align == "right":
        return (" " * pad) + text
    if align == "center":
        left = pad // 2
        right = pad - left
        return (" " * left) + text + (" " * right)
    return text + (" " * pad)


def parse_hex_color(color: str) -> Optional[Tuple[int, int, int]]:
    color = color.strip()
    if not color.startswith("#"):
        return None
    hexval = color[1:]
    if len(hexval) == 3:
        hexval = "".join([c * 2 for c in hexval])
    if len(hexval) != 6:
        return None
    try:
        r = int(hexval[0:2], 16)
        g = int(hexval[2:4], 16)
        b = int(hexval[4:6], 16)
    except ValueError:
        return None
    return r, g, b


def color_to_ansi(color: str, background: bool = False) -> str:
    if not color:
        return ""
    color = color.strip()
    if color.startswith("color(") and color.endswith(")"):
        try:
            idx = int(color[6:-1])
            return f"\x1b[{48 if background else 38};5;{idx}m"
        except ValueError:
            return ""
    rgb = parse_hex_color(color)
    if rgb:
        r, g, b = rgb
        return f"\x1b[{48 if background else 38};2;{r};{g};{b}m"
    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (220, 50, 47),
        "green": (133, 153, 0),
        "yellow": (181, 137, 0),
        "blue": (38, 139, 210),
        "magenta": (211, 54, 130),
        "cyan": (42, 161, 152),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }
    rgb = named.get(color.lower())
    if rgb:
        r, g, b = rgb
        return f"\x1b[{48 if background else 38};2;{r};{g};{b}m"
    return ""


def style_text(text: str, fg: str = "", bg: str = "", bold: bool = False) -> str:
    parts = []
    fg_code = color_to_ansi(fg, background=False) if fg else ""
    bg_code = color_to_ansi(bg, background=True) if bg else ""
    if bold:
        parts.append(ANSI_BOLD)
    if fg_code:
        parts.append(fg_code)
    if bg_code:
        parts.append(bg_code)
    if parts:
        return "".join(parts) + text + ANSI_RESET
    return text


def join_lr(left: str, right: str, width: int) -> str:
    left = left or ""
    right = right or ""
    total = visible_len(left) + visible_len(right)
    if total + 1 > width:
        if visible_len(right) >= width:
            return crop_ansi(right, width)
        available = max(1, width - visible_len(right) - 1)
        left = crop_ansi(left, available)
    space = width - visible_len(left) - visible_len(right)
    if space < 1:
        space = 1
    return left + (" " * space) + right


def apply_padding(lines: List[str], width: int, pad_v: int, pad_h: int) -> List[str]:
    content_width = max(0, width - (pad_h * 2))
    padded = [" " * width for _ in range(pad_v)]
    for line in lines:
        line = pad_ansi(line, content_width)
        padded.append((" " * pad_h) + line + (" " * pad_h))
    padded.extend([" " * width for _ in range(pad_v)])
    return padded


def clamp_width(text: str, width: int, align: str = "left") -> str:
    return pad_ansi(text, width, align=align)


class FixtureEngine:
    def __init__(self, fixture_path: str):
        self.path = fixture_path
        self.data = self._load(fixture_path)
        self.scenarios: Dict[str, Dict[str, Any]] = dict(self.data.get("scenarios", {}) or {})

    def _load(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def scenario_names(self) -> List[str]:
        return sorted(self.scenarios.keys())

    def get(self, name: str) -> Dict[str, Any]:
        return dict(self.scenarios.get(name, {}) or {})


def load_config(path: str) -> Dict[str, Any]:
    cfg = read_yaml(path)

    includes = cfg.get("includes", {}) or {}
    for section, inc_path in includes.items():
        inc_path = str(inc_path or "").strip()
        if not inc_path:
            continue
        block = read_yaml(inc_path)
        if section == "all":
            cfg = deep_merge(cfg, block)
            continue
        existing = cfg.get(section, {})
        if not isinstance(existing, dict):
            existing = {}
        cfg[section] = deep_merge(existing, block)

    cfg.setdefault("layout", {})
    cfg.setdefault("defaults", {})
    cfg.setdefault("header", {})
    cfg.setdefault("footer", {})
    cfg.setdefault("climate", {})
    cfg.setdefault("middle", {})
    cfg.setdefault("right_panel", {})
    cfg.setdefault("ups_alarm", {})
    cfg.setdefault("ui", {})

    cfg["layout"].setdefault("columns", 3)

    d = cfg["defaults"]
    d.setdefault("color_normal", "#ffb000")
    d.setdefault("color_normal_dim", "#aa7700")
    d.setdefault("color_on", "#468a00")
    d.setdefault("color_on_bright", "#88ff55")
    d.setdefault("color_warn_a", "red")
    d.setdefault("color_warn_b", "white")
    d.setdefault("color_border", d.get("color_normal", "#ffb000"))

    d.setdefault("alarm_text_a", "white")
    d.setdefault("alarm_bg_a", "red")
    d.setdefault("alarm_text_b", "red")
    d.setdefault("alarm_bg_b", "white")

    d.setdefault("enable_alarm_sound", False)
    d.setdefault("alarm_sound_path", "/usr/share/sounds/alsa/Front_Center.wav")

    d.setdefault("poll_seconds", 2.0)
    d.setdefault("history_refresh_seconds", 60)
    d.setdefault("daily_avg", True)

    cfg["header"].setdefault("title", "HA STATUS BOARD")
    cfg["header"].setdefault("right", [])
    cfg["footer"].setdefault("right", [])

    cfg["climate"].setdefault("floors", [])
    if not cfg["climate"]["floors"]:
        raise ValueError("Config climate.floors is required.")

    cfg["middle"].setdefault("boiler_entity", "switch.boiler")
    cfg["middle"].setdefault("boiler_on_state", "on")

    cfg["ups_alarm"].setdefault("entity", "sensor.smartups_status")
    cfg["ups_alarm"].setdefault("alarm_contains", ["on battery", "discharging"])

    ui = cfg["ui"]
    ui.setdefault("panel_padding", [1, 2])
    ui.setdefault("panel_margin_x", 1)
    ui.setdefault("screen_margin_y", 0)

    return cfg


class HatuiAnsiApp:
    def __init__(self, ha_url: str, ha_token: str, cfg_path: str):
        self.cfg = load_config(cfg_path)

        d = self.cfg["defaults"]
        self.color_normal = str(d["color_normal"])
        self.color_normal_dim = str(d["color_normal_dim"])
        self.color_on = str(d["color_on"])
        self.color_on_bright = str(d["color_on_bright"])
        self.color_warn_a = str(d["color_warn_a"])
        self.color_warn_b = str(d["color_warn_b"])
        self.color_border = str(d["color_border"])

        self.alarm_text_a = str(d["alarm_text_a"])
        self.alarm_bg_a = str(d["alarm_bg_a"])
        self.alarm_text_b = str(d["alarm_text_b"])
        self.alarm_bg_b = str(d["alarm_bg_b"])
        self.enable_alarm_sound = bool(d["enable_alarm_sound"])
        self.alarm_sound_path = str(d["alarm_sound_path"])

        self.poll_seconds = float(d["poll_seconds"])
        self.history_refresh_seconds = int(d["history_refresh_seconds"])
        self.daily_avg = bool(d["daily_avg"])

        ui = self.cfg.get("ui", {})
        pad_v, pad_h = ui.get("panel_padding", [1, 2])
        self.panel_pad_v = int(pad_v)
        self.panel_pad_h = int(pad_h)
        self.panel_margin_x = int(ui.get("panel_margin_x", 1))
        self.screen_margin_y = int(ui.get("screen_margin_y", 0))

        self.ha = HAClient(ha_url, ha_token)

        self.columns = max(1, int(self.cfg["layout"].get("columns", 3)))

        self._session: Optional[aiohttp.ClientSession] = None

        self._spinner_i = 0
        self._flash = False
        self._pulse = False
        self._alarm_prev_should_flash = False
        self._should_flash = False

        self.last_ok: Optional[datetime] = None
        self.last_err: Optional[str] = None

        self.entity_states: Dict[str, EntityState] = {}
        self.climate_avg: Dict[str, Optional[float]] = {}
        self.boiler_bar_fracs: Optional[List[float]] = None
        self.boiler_bar_label: Optional[str] = None
        self.boiler_bar_tick: Optional[str] = None
        self.boiler_last_on_end: Optional[datetime] = None
        self.boiler_on_live: bool = False

        self.runtime_overrides: Dict[str, EntityState] = {}
        self.flash_mode = "auto"  # auto|on|off
        self.right_panel_history: Dict[str, List[Dict[str, Any]]] = {}

        self.fixtures: Optional[FixtureEngine] = None
        if FIXTURE_PATH and os.path.exists(FIXTURE_PATH):
            self.fixtures = FixtureEngine(FIXTURE_PATH)

        self._ctl_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self.ups_alarm = False
        self._stop = asyncio.Event()

    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        sys.stdout.write(ANSI_HIDE_CURSOR)
        sys.stdout.flush()
        self.start_control_pipe()
        tasks = [
            asyncio.create_task(self.control_consumer()),
            asyncio.create_task(self.poll_fast_loop()),
            asyncio.create_task(self.poll_history_loop()),
            asyncio.create_task(self.heartbeat_loop()),
            asyncio.create_task(self.flash_loop()),
            asyncio.create_task(self.render_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self._stop.set()
            for task in tasks:
                task.cancel()
            if self._session:
                await self._session.close()
            sys.stdout.write(ANSI_SHOW_CURSOR)
            sys.stdout.write(ANSI_RESET)
            sys.stdout.flush()

    async def poll_fast_loop(self) -> None:
        while not self._stop.is_set():
            await self.poll_fast()
            await asyncio.sleep(self.poll_seconds)

    async def poll_history_loop(self) -> None:
        while not self._stop.is_set():
            await self.poll_history()
            await asyncio.sleep(self.history_refresh_seconds)

    async def heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self.heartbeat()
            await asyncio.sleep(0.2)

    async def flash_loop(self) -> None:
        while not self._stop.is_set():
            self.toggle_flash_and_pulse()
            await asyncio.sleep(1.0)

    async def render_loop(self) -> None:
        while not self._stop.is_set():
            self.render_screen()
            await asyncio.sleep(0.2)

    def toggle_flash_and_pulse(self) -> None:
        self._flash = not self._flash
        self._pulse = not self._pulse

    def play_alarm_sound(self) -> None:
        if not self.enable_alarm_sound:
            return
        try:
            subprocess.Popen(
                ["aplay", "-q", self.alarm_sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log(f"alarm sound failed: {e}")

    def start_control_pipe(self) -> None:
        try:
            d = os.path.dirname(CTL_PATH) or "."
            os.makedirs(d, exist_ok=True)

            if os.path.exists(CTL_PATH):
                st = os.stat(CTL_PATH)
                if not stat.S_ISFIFO(st.st_mode):
                    raise RuntimeError(f"{CTL_PATH} exists but is not a FIFO")
            else:
                os.mkfifo(CTL_PATH, 0o666)

            loop = asyncio.get_running_loop()

            def reader_thread() -> None:
                while True:
                    try:
                        with open(CTL_PATH, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                asyncio.run_coroutine_threadsafe(self._ctl_queue.put(line), loop)
                    except Exception as e:
                        log(f"CTL reader error: {e}")
                        import time
                        time.sleep(0.5)

            t = threading.Thread(target=reader_thread, daemon=True)
            t.start()
            log(f"CTL FIFO ready: {CTL_PATH}")

        except Exception as e:
            log(f"Failed to start control pipe: {e}")

    async def control_consumer(self) -> None:
        while not self._stop.is_set():
            cmd = await self._ctl_queue.get()
            try:
                self.apply_command(cmd)
            except Exception as e:
                log(f"Bad command '{cmd}': {e}")

    def apply_command(self, cmd: str) -> None:
        c = cmd.strip()
        for keyword in ("set", "clear", "flash", "scenario"):
            if c.lower().startswith(f"{keyword}:"):
                c = f"{keyword} {c[len(keyword) + 1:]}"
                break

        if c.lower().startswith("set "):
            rest = c[4:].strip()
            if "=" not in rest:
                raise ValueError("set requires <entity_id>=<state>")
            eid, val = rest.split("=", 1)
            self.runtime_overrides[eid.strip()] = EntityState(
                state=val.strip(),
                attributes={},
                last_changed=datetime.now(timezone.utc).isoformat(),
            )
            return

        if c.lower().startswith("clear "):
            eid = c[6:].strip()
            self.runtime_overrides.pop(eid, None)
            return

        if c.lower() == "clear_all":
            self.runtime_overrides.clear()
            return

        if c.lower().startswith("flash "):
            mode = c[6:].strip().lower()
            if mode not in ("auto", "on", "off"):
                raise ValueError("flash must be auto|on|off")
            self.flash_mode = mode
            return

        if c.lower().startswith("scenario "):
            name = c[9:].strip()
            if not self.fixtures:
                raise ValueError("No fixtures loaded (create fixtures.yaml)")
            block = self.fixtures.get(name)
            if not block:
                raise ValueError(f"Unknown scenario: {name}")
            for eid, val in block.items():
                self.runtime_overrides[str(eid)] = EntityState(
                    state=str(val),
                    attributes={},
                    last_changed=datetime.now(timezone.utc).isoformat(),
                )
            return

        raise ValueError("Unknown command")

    def format_items(self, items: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for it in items:
            eid = str(it.get("id", "")).strip()
            label = str(it.get("label", "")).strip()
            fmt = str(it.get("format", "{state}")).strip()
            st = self.entity_states.get(eid)
            state_str = st.state if st else "-"
            attrs = st.attributes if st else {}
            try:
                txt = fmt.format(state=state_str, **attrs)
            except Exception:
                txt = state_str
            parts.append(f"{label}:{txt}" if label else str(txt))
        return "  ".join(parts)

    def right_panel_blocks(self) -> List[Dict[str, Any]]:
        rp = self.cfg.get("right_panel", {}) or {}
        return list(rp.get("blocks", []) or [])

    def right_panel_label_width(self) -> int:
        rp = self.cfg.get("right_panel", {}) or {}
        return int(rp.get("label_width", 12))

    def right_panel_bar_width(self) -> int:
        rp = self.cfg.get("right_panel", {}) or {}
        return int(rp.get("bar_width", 24))

    def right_panel_timeline_width(self) -> int:
        rp = self.cfg.get("right_panel", {}) or {}
        return int(rp.get("timeline_width", 32))

    def collect_right_panel_entities(self) -> List[str]:
        entity_ids: List[str] = []
        for block in self.right_panel_blocks():
            btype = str(block.get("type", "")).strip().lower()
            if btype == "value_row":
                for key in ("left", "right"):
                    info = block.get(key, {}) or {}
                    entity_ids.append(str(info.get("entity", "")).strip())
            elif btype == "ranked_list":
                for item in block.get("items", []) or []:
                    entity_ids.append(str(item.get("entity", "")).strip())
            elif btype == "stacked_bar":
                entity_ids.append(str(block.get("total_entity", "")).strip())
                for seg in block.get("segments", []) or []:
                    entity_ids.append(str(seg.get("entity", "")).strip())
            elif btype == "timeline_onoff":
                for item in block.get("items", []) or []:
                    entity, _, _ = parse_active_when(item.get("active_when", ""))
                    entity_ids.append(str(entity).strip())
        return [eid for eid in entity_ids if eid]

    def collect_right_panel_history_entities(self) -> Dict[str, float]:
        entity_windows: Dict[str, float] = {}
        for block in self.right_panel_blocks():
            btype = str(block.get("type", "")).strip().lower()
            if btype != "timeline_onoff":
                continue
            window_hours = float(block.get("window_hours", 24))
            for item in block.get("items", []) or []:
                entity, _, _ = parse_active_when(item.get("active_when", ""))
                entity = str(entity).strip()
                if not entity:
                    continue
                prev = entity_windows.get(entity, 0.0)
                if window_hours > prev:
                    entity_windows[entity] = window_hours
        return entity_windows

    def heartbeat(self) -> None:
        self._spinner_i = (self._spinner_i + 1) % len(SPINNER_FRAMES)
        if self.flash_mode == "on":
            should_flash = True
        elif self.flash_mode == "off":
            should_flash = False
        else:
            should_flash = bool(self.ups_alarm or self.last_err)
        self._should_flash = should_flash

        if should_flash and not self._alarm_prev_should_flash:
            self.play_alarm_sound()
        self._alarm_prev_should_flash = should_flash

    async def poll_fast(self) -> None:
        try:
            if not self._session:
                return

            entity_ids: List[str] = []

            for it in self.cfg["header"].get("right", []):
                entity_ids.append(str(it.get("id", "")).strip())
            for it in self.cfg["footer"].get("right", []):
                entity_ids.append(str(it.get("id", "")).strip())

            mid_cfg = self.cfg.get("middle", {}) or {}
            boiler_id = str(mid_cfg.get("boiler_entity", "")).strip()
            if boiler_id:
                entity_ids.append(boiler_id)

            for floor in self.cfg["climate"]["floors"]:
                for room in floor.get("rooms", []):
                    entity_ids.append(str(room.get("id", "")).strip())
                    heat_eid = str(room.get("heat_entity", "")).strip()
                    if heat_eid:
                        entity_ids.append(heat_eid)
                    set_eid = str(room.get("set_entity", "")).strip()
                    if set_eid:
                        entity_ids.append(set_eid)

            entity_ids.extend(self.collect_right_panel_entities())

            entity_ids.extend(list(self.runtime_overrides.keys()))

            seen = set()
            entity_ids = [x for x in entity_ids if x and not (x in seen or seen.add(x))]

            results = await asyncio.gather(*[self.ha.get_state(self._session, eid) for eid in entity_ids])
            for eid, st in zip(entity_ids, results):
                self.entity_states[eid] = st

            for eid, st in self.runtime_overrides.items():
                self.entity_states[eid] = st

            if boiler_id:
                boiler_on_state = str(mid_cfg.get("boiler_on_state", "on")).strip().lower()
                boiler_state_live = (self.entity_states.get(boiler_id).state if self.entity_states.get(boiler_id) else "").strip().lower()
                self.boiler_on_live = (boiler_state_live == boiler_on_state)

            ua = self.cfg.get("ups_alarm", {}) or {}
            ups_id = str(ua.get("entity", "sensor.smartups_status")).strip()
            ups_state = (self.entity_states.get(ups_id).state if self.entity_states.get(ups_id) else "").lower()
            phrases = [str(p).lower() for p in (ua.get("alarm_contains", ["on battery", "discharging"]) or [])]
            self.ups_alarm = any(p in ups_state for p in phrases)

            self.last_ok = datetime.now(timezone.utc)
            self.last_err = None

        except Exception as e:
            self.last_err = str(e)
            log("poll_fast error: " + str(e))
            log(traceback.format_exc())

    async def poll_history(self) -> None:
        try:
            if not self._session:
                return

            now = datetime.now(timezone.utc)

            mid_cfg = self.cfg.get("middle", {}) or {}
            boiler_id = str(mid_cfg.get("boiler_entity", "")).strip()
            boiler_on_state = str(mid_cfg.get("boiler_on_state", "on")).strip().lower()
            if boiler_id:
                hist = await self.ha.get_history(self._session, boiler_id, now - timedelta(hours=12))
                periods = 46  # fixed width within left panel
                fracs, last_on_end, current_on_from_history = compute_period_fractions(
                    hist,
                    now,
                    on_states=[boiler_on_state, "on", "true", "heating"],
                    window_hours=12.0,
                    periods=periods,
                )

                boiler_state_live = (self.entity_states.get(boiler_id).state if self.entity_states.get(boiler_id) else "").strip().lower()
                boiler_on_live = (boiler_state_live == boiler_on_state)
                self.boiler_on_live = boiler_on_live

                now_local = datetime.now().astimezone()
                end_local = now_local
                start_local = end_local - timedelta(hours=12)
                mid_local = start_local + timedelta(hours=6)

                t_start = start_local.strftime("%H:%M")
                t_mid = mid_local.strftime("%H:%M")
                t_end = end_local.strftime("%H:%M")

                width = periods
                label = ["·"] * width

                def put(pos: int, s: str) -> None:
                    pos = max(0, min(width - len(s), int(pos)))
                    for i, ch in enumerate(s):
                        label[pos + i] = ch

                put(0, t_start)
                put((width // 2) - (len(t_mid) // 2), t_mid)
                put(width - len(t_end), t_end)
                label_line = "".join(label).replace("·", " ")

                tick = ["─"] * width
                for p in (0, width // 2, width - 1):
                    tick[p] = "│"
                tick_line = "".join(tick)

                self.boiler_bar_fracs = fracs
                self.boiler_bar_label = label_line
                self.boiler_bar_tick = tick_line
                self.boiler_last_on_end = last_on_end

            else:
                self.boiler_bar_fracs = None
                self.boiler_bar_label = None
                self.boiler_bar_tick = None
                self.boiler_last_on_end = None

            if self.daily_avg:
                local_now = datetime.now().astimezone()
                local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
                start_day = local_midnight.astimezone(timezone.utc)

                climate_ids: List[str] = []
                for floor in self.cfg["climate"]["floors"]:
                    for room in floor.get("rooms", []):
                        rid = str(room.get("id", "")).strip()
                        if rid:
                            climate_ids.append(rid)

                for eid in climate_ids:
                    hist = await self.ha.get_history(self._session, eid, start_day)
                    vals2 = [safe_float(r.get("state")) for r in hist]
                    vals = [v for v in vals2 if v is not None]
                    self.climate_avg[eid] = (sum(vals) / len(vals)) if vals else None

            self.right_panel_history.clear()
            entity_windows = self.collect_right_panel_history_entities()
            for entity_id, window_hours in entity_windows.items():
                try:
                    hist = await self.ha.get_history(
                        self._session,
                        entity_id,
                        now - timedelta(hours=window_hours),
                    )
                    self.right_panel_history[entity_id] = hist
                except Exception as e:
                    log(f"right_panel history error for {entity_id}: {e}")

        except Exception as e:
            self.last_err = str(e)
            log("poll_history error: " + str(e))
            log(traceback.format_exc())

    def render_screen(self) -> None:
        term = shutil.get_terminal_size(fallback=(120, 40))
        width = max(40, term.columns)
        height = max(20, term.lines)

        header_lines = self.render_header(width)
        footer_lines = self.render_footer(width)

        content_height = height - len(header_lines) - len(footer_lines)
        if content_height < 1:
            content_height = 1

        panel_lines = self.render_panels(width, content_height)

        lines = []
        if self.screen_margin_y > 0:
            lines.extend(["" for _ in range(self.screen_margin_y)])
        lines.extend(header_lines)
        lines.extend(panel_lines)
        lines.extend(footer_lines)
        if self.screen_margin_y > 0:
            lines.extend(["" for _ in range(self.screen_margin_y)])

        output = ANSI_CLEAR + "\n".join(lines[:height])
        sys.stdout.write(output)
        sys.stdout.flush()

    def render_header(self, width: int) -> List[str]:
        spin = SPINNER_FRAMES[self._spinner_i]
        now_local = datetime.now().strftime("%a %d-%b-%Y %H:%M:%S")

        title = str(self.cfg["header"].get("title", "HA STATUS BOARD"))
        left = f"MODE:LIVE v{__version__}  {title}  {spin}  {now_local}"
        right = self.format_items(self.cfg["header"].get("right", []))
        line = join_lr(left, right, width)

        if self._should_flash:
            fg = self.alarm_text_a if self._flash else self.alarm_text_b
            bg = self.alarm_bg_a if self._flash else self.alarm_bg_b
            line = style_text(line, fg=fg, bg=bg, bold=True)
        else:
            line = style_text(line, fg=self.color_normal)
        return [line]

    def render_footer(self, width: int) -> List[str]:
        conn = "OK" if self.last_ok else "-"
        if self.last_err:
            conn = "ERROR"
        if self.last_ok:
            age = int((datetime.now(timezone.utc) - self.last_ok).total_seconds())
            ok_text = f"Last OK: {age}s"
        else:
            ok_text = "Last OK: -"
        err_text = (self.last_err or "-")[:70]
        left_f = f"Conn: {conn}  {ok_text}  Err: {err_text}"
        right_f = self.format_items(self.cfg["footer"].get("right", []))
        line = join_lr(left_f, right_f, width)
        line = style_text(line, fg=self.color_normal_dim)
        return [line]

    def render_panels(self, width: int, height: int) -> List[str]:
        columns = max(1, self.columns)
        gap = max(0, self.panel_margin_x)
        total_gap = gap * (columns - 1)
        panel_width = max(10, (width - total_gap) // columns)

        panel_sets: List[List[str]] = []
        left = self.render_climate_panel(panel_width)
        panel_sets.append(left)

        if columns >= 2:
            panel_sets.append(self.render_middle_panel(panel_width))
        if columns >= 3:
            panel_sets.append(self.render_right_panel(panel_width))

        panel_sets = [apply_padding(lines, panel_width, self.panel_pad_v, self.panel_pad_h) for lines in panel_sets]

        max_lines = max(len(p) for p in panel_sets) if panel_sets else 0
        max_lines = min(max_lines, height)

        lines = []
        for i in range(max_lines):
            row_parts = []
            for p in panel_sets:
                content = p[i] if i < len(p) else ""
                row_parts.append(clamp_width(content, panel_width))
            lines.append((" " * gap).join(row_parts))
        if len(lines) < height:
            lines.extend(["" for _ in range(height - len(lines))])
        return lines

    def render_middle_panel(self, width: int) -> List[str]:
        lines = [
            style_text("STATUS", fg=self.color_normal),
            "",
            style_text("Live control FIFO:", fg=self.color_normal_dim),
            style_text(f"  {CTL_PATH}", fg=self.color_normal_dim),
            "",
            style_text("Examples:", fg=self.color_normal_dim),
            style_text("  set sensor.smartups_status=On Battery, Battery Discharging", fg=self.color_normal_dim),
            style_text("  clear sensor.smartups_status", fg=self.color_normal_dim),
            style_text("  scenario ups_alarm", fg=self.color_normal_dim),
            style_text("  flash auto|on|off", fg=self.color_normal_dim),
        ]
        return lines

    def get_entity_state_value(self, entity_id: str) -> Tuple[str, Optional[float], Dict[str, Any]]:
        entity_id = str(entity_id or "").strip()
        st = self.entity_states.get(entity_id)
        state_str = st.state if st else ""
        attrs = st.attributes if st else {}
        value = safe_float(state_str) if state_str else None
        return state_str, value, attrs

    def format_entity_value(
        self,
        fmt: str,
        state_str: str,
        value: Optional[float],
        attrs: Dict[str, Any],
    ) -> str:
        fmt = str(fmt or "{state}")
        if value is None:
            if "{state" in fmt:
                try:
                    return fmt.format(state=state_str, **attrs)
                except Exception:
                    return state_str or "—"
            return "—"
        try:
            return fmt.format(value=value, state=state_str, **attrs)
        except Exception:
            return state_str or "—"

    def render_value_cell(self, cfg: Dict[str, Any]) -> str:
        label = str(cfg.get("label", "")).strip()
        entity_id = str(cfg.get("entity", "")).strip()
        fmt = str(cfg.get("fmt", "{value}"))
        state_str, value, attrs = self.get_entity_state_value(entity_id)
        value_text = self.format_entity_value(fmt, state_str, value, attrs)
        parts = []
        if label:
            parts.append(style_text(label + " ", fg=self.color_normal_dim))
        parts.append(style_text(value_text, fg=self.color_normal))
        return "".join(parts)

    def render_value_row(self, block: Dict[str, Any], width: int) -> List[str]:
        left_cfg = block.get("left", {}) or {}
        right_cfg = block.get("right", {}) or {}
        left_text = self.render_value_cell(left_cfg)
        right_text = self.render_value_cell(right_cfg)
        line = join_lr(left_text, right_text, width)
        return [line]

    def render_ranked_list(self, block: Dict[str, Any], width: int) -> List[str]:
        title = str(block.get("title", "")).strip()
        limit = int(block.get("limit", 5))
        value_fmt = str(block.get("value_fmt", "{value:,.0f} W"))
        thresholds = block.get("thresholds", {}) or {}
        red_thr = safe_float(thresholds.get("red"))
        yellow_thr = safe_float(thresholds.get("yellow"))
        label_w = self.right_panel_label_width()

        rows: List[Tuple[str, Optional[float]]] = []
        for item in block.get("items", []) or []:
            name = str(item.get("name", item.get("entity", ""))).strip()
            entity_id = str(item.get("entity", "")).strip()
            state_str, value, _ = self.get_entity_state_value(entity_id)
            if value is None:
                value = safe_float(state_str)
            rows.append((name, value))

        rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0.0)))
        rows = rows[:max(0, limit)]

        lines: List[str] = []
        if title:
            lines.append(style_text(title, fg=self.color_normal))

        for name, value in rows:
            if value is None:
                value_text = "—"
                value_style = self.color_normal_dim
            else:
                try:
                    value_text = value_fmt.format(value=value)
                except Exception:
                    value_text = f"{value}"
                if red_thr is not None and value >= red_thr:
                    value_style = self.color_warn_a
                elif yellow_thr is not None and value >= yellow_thr:
                    value_style = self.color_warn_b
                else:
                    value_style = self.color_normal

            left = style_text(pad_ansi(name, label_w), fg=self.color_normal)
            right = style_text(value_text, fg=value_style)
            line = join_lr(left, right, width)
            lines.append(line)
        return lines

    def render_stacked_bar(self, block: Dict[str, Any], width: int) -> List[str]:
        title = str(block.get("title", "")).strip()
        total_entity = str(block.get("total_entity", "")).strip()
        total_fmt = str(block.get("total_fmt", "{value:,.0f} W"))
        max_w = safe_float(block.get("max_w"))

        total_state, total_value, total_attrs = self.get_entity_state_value(total_entity)

        segments_cfg = block.get("segments", []) or []
        segments: List[Tuple[str, float, Optional[float]]] = []
        for seg in segments_cfg:
            name = str(seg.get("name", seg.get("entity", ""))).strip()
            entity_id = str(seg.get("entity", "")).strip()
            state_str, value, _ = self.get_entity_state_value(entity_id)
            if value is None:
                value = safe_float(state_str)
            segments.append((name, value or 0.0, value))

        seg_sum = sum(v for _, v, _ in segments)
        if total_value is None:
            total_value = seg_sum

        other_cfg = block.get("other", {}) or {}
        if str(other_cfg.get("mode", "")).strip().lower() == "remainder":
            remainder = max(0.0, (total_value or 0.0) - seg_sum)
            if remainder > 0:
                segments.append((str(other_cfg.get("name", "Other")), remainder, remainder))

        denom = max_w if max_w and max_w > 0 else (total_value or 0.0)
        bar_width = max(4, self.right_panel_bar_width())
        total_width = 0
        if denom > 0 and total_value:
            total_width = min(bar_width, int(round((total_value / denom) * bar_width)))
        total_width = max(0, total_width)

        seg_widths = [0 for _ in segments]
        if total_width > 0 and total_value and total_value > 0:
            for i, (_, v, _) in enumerate(segments):
                seg_widths[i] = int(round((v / total_value) * total_width))
            while sum(seg_widths) > total_width:
                idx = max(range(len(seg_widths)), key=lambda i: seg_widths[i])
                if seg_widths[idx] > 0:
                    seg_widths[idx] -= 1
                else:
                    break
            while sum(seg_widths) < total_width:
                idx = max(range(len(seg_widths)), key=lambda i: segments[i][1])
                seg_widths[idx] += 1

        palette = [
            self.color_on,
            self.color_normal,
            self.color_normal_dim,
            self.color_warn_b,
            self.color_warn_a,
        ]

        bar_parts: List[str] = []
        for i, width_i in enumerate(seg_widths):
            if width_i <= 0:
                continue
            color = palette[i % len(palette)]
            bar_parts.append(style_text("█" * width_i, fg=color))
        if total_width < bar_width:
            bar_parts.append(style_text("·" * (bar_width - total_width), fg=self.color_normal_dim))
        bar_line = "".join(bar_parts)

        lines: List[str] = []
        if title or total_entity:
            total_text = self.format_entity_value(total_fmt, total_state, total_value, total_attrs)
            header_line = join_lr(
                style_text(title or "Distribution", fg=self.color_normal),
                style_text(total_text, fg=self.color_normal),
                width,
            )
            lines.append(header_line)

        lines.append(bar_line)

        if block.get("show_legend", True):
            label_w = self.right_panel_label_width()
            for i, (name, _, raw_value) in enumerate(segments):
                color = palette[i % len(palette)]
                if raw_value is None:
                    val_text = "—"
                else:
                    val_text = self.format_entity_value(total_fmt, str(raw_value), raw_value, {})
                left = style_text(pad_ansi(name, label_w), fg=color)
                right = style_text(val_text, fg=color)
                lines.append(join_lr(left, right, width))
        return lines

    def compress_bool_blocks(self, blocks: List[bool], target_len: int) -> List[bool]:
        if target_len <= 0:
            return []
        if len(blocks) <= target_len:
            return blocks
        out: List[bool] = []
        step = len(blocks) / float(target_len)
        for i in range(target_len):
            start = int(i * step)
            end = int((i + 1) * step)
            if end <= start:
                end = start + 1
            chunk = blocks[start:end]
            out.append(any(chunk))
        return out

    def render_timeline_onoff(self, block: Dict[str, Any], width: int) -> List[str]:
        title = str(block.get("title", "")).strip()
        window_hours = float(block.get("window_hours", 24))
        resolution = int(block.get("resolution", 96))
        bar_width = min(resolution, self.right_panel_timeline_width())
        now = datetime.now(timezone.utc)
        label_w = self.right_panel_label_width()

        lines: List[str] = []
        if title:
            lines.append(style_text(title, fg=self.color_normal))

        for item in block.get("items", []) or []:
            name = str(item.get("name", "")).strip()
            entity, op, threshold = parse_active_when(item.get("active_when", ""))
            entity = str(entity).strip()
            history = self.right_panel_history.get(entity, []) if entity else []

            if op and threshold is not None:

                def predicate(state):
                    val = safe_float(state)
                    if val is None:
                        return False
                    if op == ">":
                        return val > threshold
                    if op == ">=":
                        return val >= threshold
                    if op == "<":
                        return val < threshold
                    if op == "<=":
                        return val <= threshold
                    if op == "==":
                        return val == threshold
                    if op == "!=":
                        return val != threshold
                    return False

            else:
                on_states = item.get("on_states", ["on", "true", "home", "open"]) or []
                on_set = set([str(s).strip().lower() for s in on_states])

                def predicate(state):
                    return str(state or "").strip().lower() in on_set

            blocks = build_bool_bar(history, now, window_hours, resolution, predicate)
            blocks = self.compress_bool_blocks(blocks, bar_width)

            bar_parts: List[str] = []
            for is_on in blocks:
                if is_on:
                    bar_parts.append(style_text("█", fg=self.color_on))
                else:
                    bar_parts.append(style_text("·", fg=self.color_normal_dim))

            if not blocks:
                bar_parts.append(style_text("—", fg=self.color_normal_dim))

            left = style_text(pad_ansi(name or entity, label_w), fg=self.color_normal)
            bar = "".join(bar_parts)
            line = join_lr(left, bar, width)
            lines.append(line)

        return lines

    def render_divider(self) -> str:
        width = self.right_panel_label_width() + self.right_panel_bar_width() + 3
        return style_text("─" * max(8, width), fg=self.color_normal_dim)

    def render_right_panel(self, width: int) -> List[str]:
        rp = self.cfg.get("right_panel", {}) or {}
        title = str(rp.get("title", "ENERGY")).strip()

        lines = [style_text(title, fg=self.color_normal), ""]

        blocks = self.right_panel_blocks()
        if not blocks:
            lines.append(style_text("No right_panel blocks configured.", fg=self.color_normal_dim))
            return lines

        for block in blocks:
            btype = str(block.get("type", "")).strip().lower()
            if btype == "value_row":
                lines.extend(self.render_value_row(block, width))
            elif btype == "ranked_list":
                lines.extend(self.render_ranked_list(block, width))
            elif btype == "stacked_bar":
                lines.extend(self.render_stacked_bar(block, width))
            elif btype == "timeline_onoff":
                lines.extend(self.render_timeline_onoff(block, width))
            elif btype == "divider":
                lines.append(self.render_divider())
            else:
                lines.append(style_text(f"Unknown block: {btype}", fg=self.color_normal_dim))
        return lines

    def render_climate_panel(self, width: int) -> List[str]:
        mid_cfg = self.cfg.get("middle", {}) or {}
        boiler_id = str(mid_cfg.get("boiler_entity", "")).strip()
        boiler_on_state = str(mid_cfg.get("boiler_on_state", "on")).strip().lower()
        boiler_state = (self.entity_states.get(boiler_id).state if boiler_id and self.entity_states.get(boiler_id) else "-").strip().lower()
        boiler_on = (boiler_state == boiler_on_state)

        lines: List[str] = []
        boiler_bar = self.build_boiler_bar_lines()
        if boiler_bar:
            lines.extend(boiler_bar)

        boiler_label = style_text("BOILER:", fg=self.color_normal)
        if boiler_on:
            boiler_style = self.color_on_bright if self._pulse else self.color_on
        else:
            boiler_style = self.color_normal
        boiler_state_txt = style_text("ON" if boiler_on else "OFF", fg=boiler_style)
        lines.append(boiler_label + boiler_state_txt)
        lines.append("")

        unit_default = str(self.cfg["climate"].get("unit_fallback", "C")).strip()

        col_widths = [18, 13, 8, 8]
        header_cells = [
            style_text(pad_ansi("ROOM", col_widths[0]), fg=self.color_normal),
            style_text(pad_ansi("NOW", col_widths[1]), fg=self.color_normal),
            style_text(pad_ansi("SET", col_widths[2]), fg=self.color_normal),
            style_text(pad_ansi("AVG", col_widths[3]), fg=self.color_normal),
        ]
        lines.append(" ".join(header_cells))
        lines.append(
            " ".join(
                [
                    style_text(pad_ansi("─" * col_widths[0], col_widths[0]), fg=self.color_normal_dim),
                    style_text(pad_ansi("─" * col_widths[1], col_widths[1]), fg=self.color_normal_dim),
                    style_text(pad_ansi("─" * col_widths[2], col_widths[2]), fg=self.color_normal_dim),
                    style_text(pad_ansi("─" * col_widths[3], col_widths[3]), fg=self.color_normal_dim),
                ]
            )
        )

        for floor in self.cfg["climate"].get("floors", []):
            for room in floor.get("rooms", []):
                name = str(room.get("name", room.get("id", ""))).strip()
                eid = str(room.get("id", "")).strip()
                heat_eid = str(room.get("heat_entity", "")).strip()
                set_eid = str(room.get("set_entity", "")).strip()
                set_attr = str(room.get("set_attr", "temperature")).strip()

                st = self.entity_states.get(eid)
                now_v = safe_float(st.state) if st else None
                unit = (st.attributes.get("unit_of_measurement") if st else None) or unit_default
                now_txt = f"{now_v:0.1f}{unit}" if now_v is not None else "-"

                heating = False
                if heat_eid:
                    hs = (self.entity_states.get(heat_eid).state if self.entity_states.get(heat_eid) else "").strip().lower()
                    heating = (hs == "heating")

                set_v: Optional[float] = None
                if set_eid:
                    ss = self.entity_states.get(set_eid)
                    if ss:
                        if isinstance(ss.attributes, dict) and set_attr in ss.attributes:
                            set_v = safe_float(ss.attributes.get(set_attr))
                        if set_v is None:
                            set_v = safe_float(ss.state)
                set_txt = f"{set_v:0.1f}{unit}" if set_v is not None else "-"

                avg_v = self.climate_avg.get(eid)
                avg_txt = f"{avg_v:0.1f}{unit}" if avg_v is not None else "-"

                if heating:
                    room_style = self.color_on_bright if self._pulse else self.color_on
                    now_style = room_style
                else:
                    room_style = self.color_normal
                    now_style = self.color_normal

                if set_v is not None and now_v is not None and now_v >= set_v:
                    if heating:
                        room_style = self.color_on
                        now_style = self.color_on
                    else:
                        room_style = self.color_normal_dim
                        now_style = self.color_normal_dim

                row_cells = [
                    style_text(pad_ansi(name, col_widths[0]), fg=room_style),
                    style_text(pad_ansi(now_txt, col_widths[1]), fg=now_style, bold=True),
                    style_text(pad_ansi(set_txt, col_widths[2]), fg=self.color_normal_dim),
                    style_text(pad_ansi(avg_txt, col_widths[3]), fg=self.color_normal_dim),
                ]
                lines.append(" ".join(row_cells))
        return lines

    def build_boiler_bar_lines(self) -> List[str]:
        if not self.boiler_bar_fracs or not self.boiler_bar_label or not self.boiler_bar_tick:
            return []

        full_thr = 0.98
        part_thr = 0.02

        now = datetime.now(timezone.utc)
        if self.boiler_last_on_end is None:
            age_sec = 10**9
        else:
            age_sec = (now - self.boiler_last_on_end).total_seconds()

        lines = []
        lines.append(style_text(self.boiler_bar_label, fg=self.color_normal_dim))
        lines.append(style_text(self.boiler_bar_tick, fg=self.color_normal_dim))

        bar_parts: List[str] = []
        for i, frac in enumerate(self.boiler_bar_fracs):
            if frac >= full_thr:
                st = self.color_on
                ch = "█"
            elif frac > part_thr:
                st = self.color_normal
                if frac < 0.50:
                    ch = "▒"
                else:
                    ch = "░"
            else:
                st = self.color_normal_dim
                ch = "░"

            if i == (len(self.boiler_bar_fracs) - 1):
                if self.boiler_on_live:
                    st = self.color_on_bright if self._pulse else self.color_on
                    ch = "█"
                else:
                    if age_sec <= 300:
                        st = self.color_on
                        ch = "█"
                    elif age_sec <= 900:
                        st = self.color_normal
                        ch = "░"
                    else:
                        st = self.color_normal_dim
                        ch = "░"

            bar_parts.append(style_text(ch, fg=st))

        lines.append("".join(bar_parts))
        return lines


def main() -> int:
    if "--version" in os.sys.argv or "-V" in os.sys.argv:
        print(f"hatui v{__version__} (build {__build__})")
        return 0

    ha_url = os.getenv("HA_URL", "").strip()
    ha_token = os.getenv("HA_TOKEN", "").strip()
    cfg_path = os.getenv("HATUI_CONFIG", "entities.yaml").strip()

    if not cfg_path or not os.path.exists(cfg_path):
        print(f"ERROR: Config not found: {cfg_path}", flush=True)
        return 2
    if not ha_url:
        print("ERROR: HA_URL not set.", flush=True)
        return 2
    if not ha_token:
        print("ERROR: HA_TOKEN not set.", flush=True)
        return 2

    try:
        asyncio.run(HatuiAnsiApp(ha_url=ha_url, ha_token=ha_token, cfg_path=cfg_path).run())
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
