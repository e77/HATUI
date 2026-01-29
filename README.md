# HATUI
A Text UI for HomeAssistant

## SSH Menu (test modes + OTA updates)
Use the SSH menu to trigger fixture scenarios, toggle flash modes, and pull updates for YAML/PY files.
You can run it directly over SSH, or expose it via a systemd service or a shell alias.

```
./hatui_menu.py
```

Example systemd unit (optional):

```
[Unit]
Description=HATUI SSH menu
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/hatui
ExecStart=/opt/hatui/hatui_menu.py
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

Example systemd unit for the main TUI (with dependency preflight):

```
[Unit]
Description=HATUI TUI
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/hatui
ExecStart=/opt/hatui/scripts/hatui-launch.sh
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

The menu writes commands to the control FIFO (`$HATUI_CTL`, default `/run/hatui/ctl`) and supports
firmware-style updates for YAML/PY files by force-syncing the local checkout to `origin/master`.
This update path discards local changes (`git reset --hard` + `git clean -fd`).
After applying updates, it restarts `hatui-wayland.service` when available; otherwise it restarts
`hatui.service` (the default `HATUI_SERVICE`).
Restarting from the menu requires passwordless sudo for `systemctl restart`. Add a sudoers rule
for the SSH user (example for `echo77it`) so the non-interactive restart can run:

```
echo77it ALL=NOPASSWD: /bin/systemctl restart hatui.service
```

If you run the Wayland unit, swap the unit name to `hatui-wayland.service`.

### Git integration setup (required for “Check for updates”)
Before using the menu’s “Check for updates” or applying OTA updates, configure the repo’s git
remote. Run the setup script once on the device:

```
./scripts/setup_git_integration.sh
```

By default, the script reads `HATUI_GIT_REMOTE` or `config/hatui_git.json` (key `remote_url`) to
set/update the `origin` remote and adds the repo to `safe.directory` when needed (root or non-owner).

## Config includes (panel-specific YAML)
You can split configuration into panel-specific YAML files and merge them at runtime using
the `includes` section in your main config. Example:

```
includes:
  header: header.yaml
  footer: footer.yaml
  climate: climate.yaml
  middle: middle.yaml
  right_panel: right_panel.yaml
  defaults: defaults.yaml
  layout: layout.yaml
  ui: ui.yaml
  ups_alarm: ups_alarm.yaml
```

Each included file is merged into the matching section, with the included values overriding
the base config.

## Right panel include (power map blocks)
Use `right_panel.yaml` (via `includes.right_panel`) to define right panel blocks optimized
for narrow TTY layouts. Example:

```yaml
right_panel:
  title: "Power Map"
  label_width: 12
  bar_width: 24
  timeline_width: 32
  blocks:
    - type: value_row
      left:  { label: "⚡ Grid", entity: "sensor.electricity_import_power", fmt: "{value:,.0f} W" }
      right: { label: "🔥 Gas",  entity: "sensor.gas_power",              fmt: "{value:.2f} kW" }

    - type: divider

    - type: ranked_list
      title: "Top consumers"
      limit: 8
      value_fmt: "{value:,.0f} W"
      thresholds:
        red: 2000
        yellow: 800
      items:
        - { name: "Oven/Hob",       entity: "sensor.power_oven" }
        - { name: "Kettle",         entity: "sensor.power_kettle" }
        - { name: "Dishwasher",     entity: "sensor.power_dishwasher" }

    - type: divider

    - type: stacked_bar
      title: "Distribution"
      total_entity: "sensor.electricity_import_power"
      max_w: 7000
      segments:
        - { name: "Kitchen", entity: "sensor.power_kitchen" }
        - { name: "Laundry", entity: "sensor.power_laundry" }
      other:
        mode: "remainder"

    - type: timeline_onoff
      title: "Last 24h activity"
      window_hours: 24
      resolution: 96
      items:
        - { name: "Oven",   active_when: "sensor.power_oven > 300" }
        - { name: "Kettle", active_when: "sensor.power_kettle > 300" }
```

Supported block types:
- `value_row`: Two headline values on one row (left/right) using `label`, `entity`, and `fmt`.
- `ranked_list`: Sorted list of items by current value, with optional `thresholds` for color.
- `stacked_bar`: Horizontal stacked bar showing distribution (optionally `other.mode: remainder`).
- `timeline_onoff`: 24h activity strips using `active_when` expressions (thresholds or state lists).
- `divider`: Horizontal separator line.

## Boot splash notes
See `docs/boot-splash.md` for a starting point on hiding the Raspberry Pi OS boot text and
showing a custom splash before HATUI starts.
