# Boot splash notes (Raspberry Pi OS)

These notes outline a simple approach to hide Raspberry Pi OS boot text and show a custom splash
before the HATUI TUI starts. They are intended as a starting point for a production boot flow.

## High-level approach
1. Disable kernel boot text (quiet boot).
2. Start a splash service that shows an image or a solid color.
3. Replace the splash with the HATUI TUI once your services are ready.

## Steps to consider
- Add `quiet loglevel=0` to `/boot/cmdline.txt`.
- Disable console cursor and blanking (optional).
- Start a systemd service that uses `fbi`, `plymouth`, or `ffmpeg` to draw a splash on `/dev/fb0`.
- Stop the splash service when `hatui-wayland.service` starts (or when Home Assistant data is ready).

## Example systemd unit (skeleton)
```
[Unit]
Description=HATUI boot splash
DefaultDependencies=no
After=local-fs.target
Before=getty@tty1.service

[Service]
Type=simple
ExecStart=/usr/bin/fbi -T 1 -d /dev/fb0 --noverbose /opt/hatui/assets/splash.png
StandardInput=tty
StandardOutput=tty
StandardError=tty
TTYPath=/dev/tty1
TTYReset=yes
TTYVHangup=yes

[Install]
WantedBy=sysinit.target
```

This service should be started early and stopped when `hatui-wayland.service` begins.
