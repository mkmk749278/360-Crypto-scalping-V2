"""Deployment commands (admin) — /deploy, /restart, /rollback."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List

from src.commands.registry import CommandContext, CommandRegistry

registry = CommandRegistry()

_REPO_ROOT: Path = Path(__file__).parent.parent.parent


@registry.command(
    "/deploy",
    aliases=["/update_code"],
    admin=True,
    group="deploy",
    help_text="Git pull (deploy latest code)",
)
async def handle_deploy(args: List[str], ctx: CommandContext) -> None:
    await ctx.reply("⏳ Running git pull …")
    try:
        result = subprocess.run(
            ["git", "pull"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_REPO_ROOT),
        )
        output = (result.stdout + result.stderr).strip() or "No output."
        await ctx.reply(f"✅ git pull result:\n```\n{output}\n```")
    except subprocess.TimeoutExpired:
        await ctx.reply("❌ git pull timed out.")
    except Exception as exc:
        await ctx.reply(f"❌ git pull error: {exc}")


@registry.command(
    "/restart",
    aliases=["/restart_engine"],
    admin=True,
    group="deploy",
    help_text="Restart the engine",
)
async def handle_restart(args: List[str], ctx: CommandContext) -> None:
    await ctx.reply("🔄 Restarting engine tasks …")
    try:
        if ctx.restart_callback:
            await ctx.restart_callback(ctx.chat_id)
        else:
            await ctx.reply("❌ Restart callback not configured.")
    except Exception as exc:
        await ctx.reply(f"❌ Restart error: {exc}")


@registry.command(
    "/rollback",
    aliases=["/rollback_code"],
    admin=True,
    group="deploy",
    help_text="Rollback to commit: /rollback <commit>",
)
async def handle_rollback(args: List[str], ctx: CommandContext) -> None:
    if not args:
        await ctx.reply("Usage: /rollback <commit>")
        return
    commit = args[0]
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9.\-]{0,79}$', commit):
        await ctx.reply("❌ Invalid commit reference.")
        return
    await ctx.reply(f"⏳ Running git checkout {commit} …")
    try:
        result = subprocess.run(
            ["git", "checkout", commit],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_REPO_ROOT),
        )
        output = (result.stdout + result.stderr).strip() or "Done."
        await ctx.reply(f"✅ Rollback result:\n```\n{output}\n```")
    except subprocess.TimeoutExpired:
        await ctx.reply("❌ git checkout timed out.")
    except Exception as exc:
        await ctx.reply(f"❌ Rollback error: {exc}")
