#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HATUI SSH Menu for test modes + OTA updates.

Usage:
  ./hatui_menu.py

Environment:
  HATUI_CTL     Path to control FIFO (default: /run/hatui/ctl)
  HATUI_FIXTURE Fixtures YAML (default: fixtures.yaml)
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import Dict, List

import yaml

CTL_PATH = os.getenv("HATUI_CTL", "/run/hatui/ctl")
FIXTURE_PATH = os.getenv("HATUI_FIXTURE", "fixtures.yaml").strip()
HATUI_SERVICE = os.getenv("HATUI_SERVICE", "hatui.service").strip()
WAYLAND_SERVICE = "hatui-wayland.service"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def read_yaml(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)


def write_ctl(command: str) -> bool:
    if not os.path.exists(CTL_PATH):
        print(f"Control FIFO not found: {CTL_PATH}")
        return False
    try:
        with open(CTL_PATH, "w", encoding="utf-8") as f:
            f.write(command.strip() + "\n")
        return True
    except Exception as exc:
        print(f"Failed to write to FIFO: {exc}")
        return False


def list_scenarios() -> List[str]:
    data = read_yaml(FIXTURE_PATH)
    scenarios = data.get("scenarios", {}) if isinstance(data, dict) else {}
    if isinstance(scenarios, dict):
        return sorted(scenarios.keys())
    return []


def git_status() -> None:
    print("\n[git status]")
    res = run(["git", "status", "-sb"])
    print(res.stdout or res.stderr)


def git_check_updates() -> List[str]:
    run(["git", "fetch", "--all", "--prune"])
    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream.returncode != 0:
        print("No upstream configured for this branch.")
        return []
    upstream_ref = upstream.stdout.strip()
    diff = run(["git", "diff", "--name-only", "HEAD.."+upstream_ref])
    files = [f for f in diff.stdout.splitlines() if f.strip()]
    print("\n[updates available]" if files else "\n[up to date]")
    for f in files:
        print(f" - {f}")
    return files


def git_apply_updates() -> bool:
    print("\n[pulling updates]")
    res = run(["git", "pull", "--ff-only"])
    print(res.stdout or res.stderr)
    return res.returncode == 0


def service_exists_or_active(unit: str) -> bool:
    if not unit:
        return False
    res = run(["systemctl", "status", "--no-pager", unit])
    return res.returncode in (0, 3)


def detect_service_unit() -> str | None:
    if service_exists_or_active(WAYLAND_SERVICE):
        return WAYLAND_SERVICE
    if service_exists_or_active(HATUI_SERVICE):
        return HATUI_SERVICE
    return None


def restart_service() -> bool:
    unit = detect_service_unit()
    if not unit:
        print("No service unit found to restart.")
        return False
    print(f"\n[restarting service] {unit}")
    res = run(["sudo", "-n", "systemctl", "restart", unit])
    print(res.stdout or res.stderr)
    if res.returncode != 0:
        print("Service restart failed.")
        print("Add this sudoers rule to allow non-interactive restart:")
        print(f"  echo77it ALL=NOPASSWD: /bin/systemctl restart {unit}")
        return False
    print("Service restarted.")
    return True


def ota_update_yaml_py() -> None:
    files = git_check_updates()
    targets = [f for f in files if f.endswith((".yaml", ".yml", ".py"))]
    if not targets:
        print("No YAML/PY updates detected.")
        return
    print("\n[update candidates]")
    for f in targets:
        print(f" - {f}")
    choice = input("Apply updates? (y/N): ").strip().lower()
    if choice == "y":
        if git_apply_updates():
            restart_service()
    else:
        print("Skipped.")


def show_menu() -> None:
    print("\nHATUI SSH MENU")
    print("1) Show git status")
    print("2) Check for updates (git fetch)")
    print("3) Apply YAML/PY updates (fast-forward)")
    print("4) Trigger scenario (fixtures)")
    print("5) Flash mode (auto/on/off)")
    print("6) Clear overrides")
    print("7) Exit")


def menu_loop() -> None:
    print(f"HATUI menu started at {datetime.now().isoformat(timespec='seconds')}")
    print(f"Control FIFO: {CTL_PATH}")
    while True:
        show_menu()
        choice = input("\nSelect: ").strip()
        if choice == "1":
            git_status()
        elif choice == "2":
            git_check_updates()
        elif choice == "3":
            ota_update_yaml_py()
        elif choice == "4":
            scenarios = list_scenarios()
            if not scenarios:
                print("No scenarios found in fixtures.")
                continue
            print("\nScenarios:")
            for i, name in enumerate(scenarios, start=1):
                print(f"{i}) {name}")
            pick = input("Select scenario: ").strip()
            try:
                idx = int(pick)
            except ValueError:
                print("Invalid selection.")
                continue
            if idx < 1 or idx > len(scenarios):
                print("Out of range.")
                continue
            cmd = f"scenario {scenarios[idx - 1]}"
            if write_ctl(cmd):
                print(f"Sent: {cmd}")
        elif choice == "5":
            mode = input("flash mode (auto/on/off): ").strip().lower()
            if mode not in ("auto", "on", "off"):
                print("Invalid mode.")
                continue
            cmd = f"flash {mode}"
            if write_ctl(cmd):
                print(f"Sent: {cmd}")
        elif choice == "6":
            if write_ctl("clear_all"):
                print("Sent: clear_all")
        elif choice == "7":
            print("Bye.")
            return
        else:
            print("Unknown option.")


def main() -> int:
    try:
        menu_loop()
        return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
