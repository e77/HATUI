#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import stat
import asyncio
import threading
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import yaml
from rich.text import Text
from rich.table import Table

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

__version__ = "9.6.6-fix2"
__build__ = "2026-01-28"

SPINNER_FRAMES = ["|", "/", "-", "\\"]

LOG_PATH = os.getenv("HATUI_LOG", "/tmp/hatui.log")
CTL_PATH = os.getenv("HATUI_CTL", "/run/hatui/ctl")
FIXTURE_PATH = os.getenv("HATUI_FIXTURE", "fixtures.yaml").strip()


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


class Panel(Static):
    def __init__(self, panel_id: str):
        super().__init__("", id=panel_id)

    def set_renderable(self, r) -> None:
        self.update(r)


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


class HatuiApp(App):
    def __init__(self, ha_url: str, ha_token: str, cfg_path: str):
        super().__init__()
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
        mx = int(ui.get("panel_margin_x", 1))
        my = int(ui.get("screen_margin_y", 0))

        self.CSS = f'''
        Screen {{ background: black; color: {self.color_normal}; }}

        #header {{
            dock: top; height: 3;
            border: round {self.color_border};
            padding: 0 1;
            margin: {my} 1 0 1;
        }}

        #footer {{
            dock: bottom; height: 3;
            border: round {self.color_border};
            padding: 0 1;
            margin: 0 1 {my} 1;
        }}

        Horizontal {{ height: 1fr; width: 100%; }}

        .col {{
            width: 1fr; height: 100%;
            border: round {self.color_border};
            padding: {int(pad_v)} {int(pad_h)};
            margin: 0 {mx};
            background: black;
        }}
        '''

        self.ha = HAClient(ha_url, ha_token)

        self.header = Static("", id="header")
        self.footer = Static("", id="footer")
        self.left_panel = Panel("left_panel")
        self.mid_panel = Panel("mid_panel")
        self.right_panel = Panel("right_panel")

        self.columns = max(1, int(self.cfg["layout"].get("columns", 3)))
        self._col_widgets: List[Static] = []

        self._session: Optional[aiohttp.ClientSession] = None

        self._spinner_i = 0
        self._flash = False
        self._pulse = False
        self._alarm_prev_should_flash = False

        self.last_ok: Optional[datetime] = None
        self.last_err: Optional[str] = None

        self.entity_states: Dict[str, EntityState] = {}
        self.climate_avg: Dict[str, Optional[float]] = {}
        self.boiler_bar: Optional[Text] = None
        self.boiler_bar_fracs: Optional[List[float]] = None
        self.boiler_bar_label: Optional[str] = None
        self.boiler_bar_tick: Optional[str] = None
        self.boiler_last_on_end: Optional[datetime] = None
        self.boiler_on_live: bool = False

        self.runtime_overrides: Dict[str, EntityState] = {}
        self.flash_mode = "auto"  # auto|on|off

        self.fixtures: Optional[FixtureEngine] = None
        if FIXTURE_PATH and os.path.exists(FIXTURE_PATH):
            self.fixtures = FixtureEngine(FIXTURE_PATH)

        self._ctl_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self.ups_alarm = False

    def compose(self) -> ComposeResult:
        yield self.header
        with Horizontal():
            for i in range(self.columns):
                yield Static("", classes="col", id=f"col{i+1}")
        yield self.footer

    async def on_mount(self) -> None:
        self._col_widgets = [self.query_one(f"#col{i+1}", Static) for i in range(self.columns)]
        if self.columns >= 1:
            self._col_widgets[0].mount(self.left_panel)
        if self.columns >= 2:
            self._col_widgets[1].mount(self.mid_panel)
        if self.columns >= 3:
            self._col_widgets[2].mount(self.right_panel)

        self._session = aiohttp.ClientSession()

        self.start_control_pipe()
        asyncio.create_task(self.control_consumer())

        self.set_interval(self.poll_seconds, self.poll_fast)
        self.set_interval(0.2, self.heartbeat)
        self.set_interval(0.5, self.toggle_flash_and_pulse)
        self.set_interval(self.history_refresh_seconds, self.poll_history)

        await self.poll_fast()
        await self.poll_history()

    async def on_unmount(self) -> None:
        try:
            if self._session:
                await self._session.close()
        except Exception:
            pass

    def toggle_flash_and_pulse(self) -> None:
        self._flash = not self._flash
        self._pulse = not self._pulse
        self.render_panels()

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
        while True:
            cmd = await self._ctl_queue.get()
            try:
                self.apply_command(cmd)
            except Exception as e:
                log(f"Bad command '{cmd}': {e}")
            self.render_panels()

    def apply_command(self, cmd: str) -> None:
        c = cmd.strip()

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

    def grid_lr(self, left: str, right: str):
        tbl = Table.grid(expand=True)
        tbl.add_column(ratio=3, justify="left", overflow="crop")
        tbl.add_column(ratio=2, justify="right", overflow="crop")
        tbl.add_row(left, right)
        return tbl

    def heartbeat(self) -> None:
        self._spinner_i = (self._spinner_i + 1) % len(SPINNER_FRAMES)
        spin = SPINNER_FRAMES[self._spinner_i]
        now_local = datetime.now().strftime("%a %d-%b-%Y %H:%M:%S")

        title = str(self.cfg["header"].get("title", "HA STATUS BOARD"))
        left = f"MODE:LIVE v{__version__}  {title}  {spin}  {now_local}"
        right = self.format_items(self.cfg["header"].get("right", []))
        self.header.update(self.grid_lr(left, right))

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
        self.footer.update(self.grid_lr(left_f, right_f))

        if self.flash_mode == "on":
            should_flash = True
        elif self.flash_mode == "off":
            should_flash = False
        else:
            should_flash = bool(self.ups_alarm or self.last_err)

        if should_flash:
            if self._flash:
                self.header.styles.color = self.alarm_text_a
                self.header.styles.background = self.alarm_bg_a
            else:
                self.header.styles.color = self.alarm_text_b
                self.header.styles.background = self.alarm_bg_b
        else:
            self.header.styles.color = self.color_normal
            self.header.styles.background = "black"

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
            self.render_panels()

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

                # Current boiler state (from live entity state) drives the final-block "now" behavior
                boiler_state_live = (self.entity_states.get(boiler_id).state if self.entity_states.get(boiler_id) else "").strip().lower()
                boiler_on_live = (boiler_state_live == boiler_on_state)
                self.boiler_on_live = boiler_on_live

                # Build axis labels (start/mid/end) with NO trailing spaces (use placeholder dots)
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
                self.boiler_bar = None

            else:
                self.boiler_bar = None
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

            self.render_panels()

        except Exception as e:
            self.last_err = str(e)
            log("poll_history error: " + str(e))
            log(traceback.format_exc())

    def render_panels(self) -> None:
        self.left_panel.set_renderable(self.render_climate_panel())
        self.mid_panel.set_renderable(self.render_middle_panel())
        self.right_panel.set_renderable(self.render_right_panel())

    def render_middle_panel(self):
        t = Text()
        t.append("STATUS\n\n", style=self.color_normal)
        t.append("Live control FIFO:\n", style=self.color_normal_dim)
        t.append(f"  {CTL_PATH}\n\n", style=self.color_normal_dim)
        t.append("Examples:\n", style=self.color_normal_dim)
        t.append("  set sensor.smartups_status=On Battery, Battery Discharging\n", style=self.color_normal_dim)
        t.append("  clear sensor.smartups_status\n", style=self.color_normal_dim)
        t.append("  scenario ups_alarm\n", style=self.color_normal_dim)
        t.append("  flash auto|on|off\n", style=self.color_normal_dim)
        return t

    def render_right_panel(self):
        t = Text()
        t.append("ENERGY\n\n", style=self.color_normal)
        t.append("v10: energy/power panel next.\n", style=self.color_normal_dim)
        if self.fixtures:
            t.append("\nSCENARIOS: ", style=self.color_normal_dim)
            t.append(", ".join(self.fixtures.scenario_names()), style=self.color_normal_dim)
        return t

    def render_climate_panel(self):
        mid_cfg = self.cfg.get("middle", {}) or {}
        boiler_id = str(mid_cfg.get("boiler_entity", "")).strip()
        boiler_on_state = str(mid_cfg.get("boiler_on_state", "on")).strip().lower()
        boiler_state = (self.entity_states.get(boiler_id).state if boiler_id and self.entity_states.get(boiler_id) else "-").strip().lower()
        boiler_on = (boiler_state == boiler_on_state)

        header = Text()
        boiler_bar = self.build_boiler_bar()
        if boiler_bar is not None:
            header.append(boiler_bar)
            header.append("\n")
        header.append("BOILER:", style=self.color_normal)
        if boiler_on:
            boiler_style = self.color_on_bright if self._pulse else self.color_on
        else:
            boiler_style = self.color_normal
        header.append("ON" if boiler_on else "OFF", style=boiler_style)
        header.append("\n\n")

        unit_default = str(self.cfg["climate"].get("unit_fallback", "C")).strip()

        tbl = Table.grid(expand=True, padding=(0, 0))
        tbl.add_column(ratio=7, justify="left", overflow="crop")
        tbl.add_column(ratio=5, justify="right", overflow="crop")
        tbl.add_column(ratio=3, justify="right", overflow="crop")
        tbl.add_column(ratio=3, justify="right", overflow="crop")

        tbl.add_row(
            Text("ROOM", style=self.color_normal),
            Text("NOW", style=self.color_normal),
            Text("SET", style=self.color_normal),
            Text("AVG", style=self.color_normal),
        )
        tbl.add_row(
            Text("─" * 18, style=self.color_normal_dim),
            Text("─" * 13, style=self.color_normal_dim),
            Text("─" * 8, style=self.color_normal_dim),
            Text("─" * 8, style=self.color_normal_dim),
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

                tbl.add_row(
                    Text(name, style=room_style),
                    Text(now_txt, style=("bold " + now_style)),
                    Text(set_txt, style=self.color_normal_dim),
                    Text(avg_txt, style=self.color_normal_dim),
                )

        outer = Table.grid(expand=True)
        outer.add_column()
        outer.add_row(header)
        outer.add_row(tbl)
        return outer

    def build_boiler_bar(self) -> Optional[Text]:
        if not self.boiler_bar_fracs or not self.boiler_bar_label or not self.boiler_bar_tick:
            return None

        # Period coloring rules:
        # - off whole period => normal_dim
        # - on part => normal
        # - on whole => on
        full_thr = 0.98
        part_thr = 0.02

        # How recently was it ON?
        now = datetime.now(timezone.utc)
        if self.boiler_last_on_end is None:
            age_sec = 10**9
        else:
            age_sec = (now - self.boiler_last_on_end).total_seconds()

        # Final (most recent) period special behavior:
        # - if ON now => on_bright blinking
        # - else if ON within last 5m => on
        # - else if ON within last 15m => normal
        # - else => normal_dim
        bar = Text()
        bar.append(Text(self.boiler_bar_label, style=self.color_normal_dim, no_wrap=True, overflow="crop"))
        bar.append("\n")
        bar.append(Text(self.boiler_bar_tick, style=self.color_normal_dim, no_wrap=True, overflow="crop"))
        bar.append("\n")

        # --- Density glyph mapping (no gaps) ---
        for i, frac in enumerate(self.boiler_bar_fracs):
            # Base style (keep your colour rules)
            if frac >= full_thr:
                st = self.color_on
                ch = "█"
            elif frac > part_thr:
                st = self.color_normal
                # 0..1 mapped to density
                if frac < 0.50:
                    ch = "▒"
                else:
                    ch = "░"
            else:
                st = self.color_normal_dim
                ch = "░"

            # Final (most recent) period special behaviour
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

            bar.append(ch, style=st)

        return bar


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
        HatuiApp(ha_url=ha_url, ha_token=ha_token, cfg_path=cfg_path).run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
