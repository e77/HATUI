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

The menu writes commands to the control FIFO (`$HATUI_CTL`, default `/run/hatui/ctl`) and supports
fast-forward `git pull` updates for YAML/PY files.

### Git integration setup (required for “Check for updates”)
Before using the menu’s “Check for updates” or applying OTA updates, configure the repo’s git
remote and upstream tracking. Run the setup script once on the device:

```
./scripts/setup_git_integration.sh
```

By default, the script reads `HATUI_GIT_REMOTE` or `config/hatui_git.json` (key `remote_url`) to
set/update the `origin` remote, ensures the current branch tracks `origin/<branch>`, and adds the
repo to `safe.directory` when needed (root or non-owner).

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

## Boot splash notes
See `docs/boot-splash.md` for a starting point on hiding the Raspberry Pi OS boot text and
showing a custom splash before HATUI starts.
