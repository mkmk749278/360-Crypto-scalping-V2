"""Telegram bot – rich signal formatting, admin commands, free/premium routing.

Uses aiohttp to call the Telegram Bot API directly (no heavy library needed).
"""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp

from config import TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_BOT_TOKEN
from src.channels.base import Signal
from src.smc import Direction
from src.utils import fmt_price, fmt_ts, get_logger

log = get_logger("telegram")


class TelegramBot:
    """Lightweight async Telegram sender + command poller."""

    def __init__(self) -> None:
        self._token = TELEGRAM_BOT_TOKEN
        self._base = f"https://api.telegram.org/bot{self._token}"
        self._session: Optional[aiohttp.ClientSession] = None
        self._offset: int = 0
        self._running = False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_message(self, chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to *chat_id*. Returns True on success."""
        if not self._token:
            log.debug("Telegram token not configured – message not sent")
            return False
        session = await self._ensure_session()
        url = f"{self._base}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                log.warning("Telegram send failed (%s): %s", resp.status, body)
        except Exception as exc:
            log.error("Telegram send error: %s", exc)
        return False

    async def send_admin_alert(self, text: str) -> bool:
        """Send a message to the admin chat."""
        if TELEGRAM_ADMIN_CHAT_ID:
            return await self.send_message(TELEGRAM_ADMIN_CHAT_ID, f"🔔 *Admin Alert*\n{text}")
        return False

    # ------------------------------------------------------------------
    # Rich signal formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_signal(sig: Signal) -> str:
        """Produce the rich, emoji-laden Telegram message for a signal."""
        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_RANGE": "⚖️",
            "360_THE_TAPE": "🐋",
        }
        emoji = chan_emojis.get(sig.channel, "📡")
        dir_emoji = "🚀" if sig.direction == Direction.LONG else "⬇️"
        dir_word = sig.direction.value

        lines = [
            f"{emoji} *{sig.channel} ALERT* 💎",
            f"Pair: *{sig.symbol}*",
            f"📈 *{dir_word}* {dir_emoji}",
            f"🚀 Entry: `{fmt_price(sig.entry)}`",
            f"🛡️ SL: `{fmt_price(sig.stop_loss)}`",
            f"🎯 TP1: `{fmt_price(sig.tp1)}` ✅",
            f"🎯 TP2: `{fmt_price(sig.tp2)}`",
        ]
        if sig.tp3 is not None:
            lines.append(f"🎯 TP3: `{fmt_price(sig.tp3)}`")
        else:
            lines.append("🎯 TP3: Dynamic/trailing")

        if sig.trailing_active:
            lines.append(f"💹 Trailing Active ({sig.trailing_desc})")

        lines.append(f"🤖 Confidence: *{sig.confidence:.0f}%*")

        sentiment_line = f"📰 AI Sentiment: {sig.ai_sentiment_label}"
        if sig.ai_sentiment_summary:
            sentiment_line += f" — {sig.ai_sentiment_summary}"
        lines.append(sentiment_line)

        lines.append(f"⚠️ Risk: {sig.risk_label}")
        lines.append(f"⏰ Time: `{fmt_ts(sig.timestamp)}`")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Admin command polling
    # ------------------------------------------------------------------

    async def poll_commands(self, handler) -> None:
        """Long-poll for admin commands (``/update_pairs``, ``/view_dashboard``, etc.)."""
        self._running = True
        while self._running:
            try:
                if not self._token:
                    await asyncio.sleep(30)
                    continue
                session = await self._ensure_session()
                url = f"{self._base}/getUpdates"
                params = {"offset": self._offset, "timeout": 20}
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(5)
                        continue
                    data = await resp.json()
                for update in data.get("result", []):
                    self._offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id == TELEGRAM_ADMIN_CHAT_ID and text.startswith("/"):
                        await handler(text, chat_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("Command poll error: %s", exc)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
