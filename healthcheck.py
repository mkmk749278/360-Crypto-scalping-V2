#!/usr/bin/env python3
"""Healthcheck — verifies the 360-Crypto-scalping-V2 engine is running and healthy."""
import os
import sys


def _engine_process_running() -> bool:
    """Return True if a Python process running src.main is found in /proc."""
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as fh:
                    cmdline = fh.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
                if "python" in cmdline and "src.main" in cmdline:
                    return True
            except (FileNotFoundError, PermissionError):
                continue
    except FileNotFoundError:
        pass
    return False


def _config_importable() -> bool:
    """Return True if the engine config module can be imported (proves deps are installed)."""
    try:
        import config
        # Access a known attribute to confirm the module loaded correctly
        _ = config.BINANCE_REST_BASE
        return True
    except Exception:
        return False


def _logs_dir_exists() -> bool:
    """Return True if the logs directory exists."""
    return os.path.isdir(os.path.join(os.path.dirname(__file__), "logs"))


if not _engine_process_running():
    print("Engine process (src.main) not found.", file=sys.stderr)
    sys.exit(1)

if not _config_importable():
    print("Config module could not be imported — dependency issue.", file=sys.stderr)
    sys.exit(1)

if not _logs_dir_exists():
    print("logs/ directory does not exist.", file=sys.stderr)
    sys.exit(1)

sys.exit(0)
