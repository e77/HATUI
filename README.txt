HATUI v9.5.0 (build 2026-01-27) — stable drop

Key points:
- ANSI/terminal rendering only (no UI frameworks).
- Left panel climate table columns: ROOM | NOW | SET | AVG
- SET (target) is optional per-room:
    set_entity: <entity_id>
    set_attr: temperature   (optional; default "temperature")
- Boiler 24h timeline bar above BOILER:ON/OFF line (uses HA history).
- UPS alarm flashes the title bar with inverted background (red/white swap).
- Optional alarm sound (defaults OFF).

Runtime control FIFO:
  sudo mkdir -p /run/hatui
  sudo mkfifo /run/hatui/ctl
  echo 'set sensor.smartups_status=On Battery, Battery Discharging' | sudo tee /run/hatui/ctl >/dev/null
  echo 'clear sensor.smartups_status' | sudo tee /run/hatui/ctl >/dev/null
  echo 'scenario ups_alarm' | sudo tee /run/hatui/ctl >/dev/null
  echo 'flash on|off|auto' | sudo tee /run/hatui/ctl >/dev/null
