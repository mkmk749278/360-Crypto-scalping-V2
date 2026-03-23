"""Telegram bot – rich signal formatting, admin commands, free/premium routing.

Uses aiohttp to call the Telegram Bot API directly (no heavy library needed).
"""

from __future__ import annotations

import asyncio
import json
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
        """Send a message to *chat_id*. Returns True on success.

        Retry behaviour:
        * **Markdown parse error** (400 + "can't parse entities"): retried once
          as plain text so the user still receives the signal.
        * **Rate limit** (429): waits ``parameters.retry_after`` seconds from
          the response body, then retries (up to 3 total attempts).
        * **Server errors** (5xx): exponential back-off (1 s, 2 s, 4 s) with up
          to 3 total attempts.
        * **Timeout**: exponential back-off, up to 3 total attempts.
        * **Other 4xx**: returned immediately as False (not recoverable).
        """
        if not self._token:
            log.debug("Telegram token not configured – message not sent")
            return False
        session = await self._ensure_session()
        url = f"{self._base}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    log.warning("Telegram send failed (%s): %s", resp.status, body)

                    # Retry as plain text if Markdown parsing failed (400 only)
                    if resp.status == 400 and "can't parse entities" in body:
                        log.info("Retrying message as plain text after Markdown parse failure")
                        plain_payload = {"chat_id": chat_id, "text": text}
                        async with session.post(
                            url, json=plain_payload, timeout=aiohttp.ClientTimeout(total=10)
                        ) as retry_resp:
                            if retry_resp.status == 200:
                                return True
                            retry_body = await retry_resp.text()
                            log.warning("Telegram plain-text retry failed (%s): %s", retry_resp.status, retry_body)
                        return False  # 400 errors are not retried further

                    # HTTP 429: rate limited – honor Retry-After from response body
                    if resp.status == 429:
                        try:
                            data = json.loads(body)
                            retry_after = float(data.get("parameters", {}).get("retry_after", 1))
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            retry_after = 1.0
                        log.info(
                            "Telegram rate limit (429) – waiting %.1fs before retry (attempt %d/%d)",
                            retry_after, attempt + 1, max_attempts,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    # HTTP 5xx: server error – exponential back-off
                    if resp.status >= 500:
                        wait = 2 ** attempt  # 1 s, 2 s, 4 s
                        log.info(
                            "Telegram server error (%d) – retrying in %ds (attempt %d/%d)",
                            resp.status, wait, attempt + 1, max_attempts,
                        )
                        await asyncio.sleep(wait)
                        continue

                    # Other 4xx: not recoverable
                    return False

            except asyncio.TimeoutError:
                wait = 2 ** attempt
                log.warning(
                    "Telegram send timeout – retrying in %ds (attempt %d/%d)",
                    wait, attempt + 1, max_attempts,
                )
                await asyncio.sleep(wait)
                continue
            except Exception as exc:
                log.error("Telegram send error: %s", exc)
                return False

        return False

    async def send_admin_alert(self, text: str) -> bool:
        """Send a message to the admin chat."""
        if TELEGRAM_ADMIN_CHAT_ID:
            return await self.send_message(TELEGRAM_ADMIN_CHAT_ID, f"🔔 *Admin Alert*\n{text}")
        return False

    async def send_photo(self, chat_id: str, photo_bytes: bytes, caption: str = "") -> bool:
        """Send a photo to *chat_id* using multipart form data. Returns True on success.

        Retry behaviour mirrors ``send_message``:
        * **Rate limit** (429): waits ``parameters.retry_after`` seconds, then retries.
        * **Server errors** (5xx): exponential back-off (1 s, 2 s, 4 s).
        * **Timeout**: exponential back-off, up to 3 total attempts.
        * **Other 4xx**: returned immediately as False.
        """
        if not self._token:
            log.debug("Telegram token not configured – photo not sent")
            return False
        session = await self._ensure_session()
        url = f"{self._base}/sendPhoto"
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                form = aiohttp.FormData()
                form.add_field("chat_id", chat_id)
                form.add_field("photo", photo_bytes, filename="chart.png", content_type="image/png")
                if caption:
                    form.add_field("caption", caption)
                    form.add_field("parse_mode", "Markdown")
                async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    log.warning("Telegram sendPhoto failed (%s): %s", resp.status, body)

                    if resp.status == 429:
                        try:
                            data = json.loads(body)
                            retry_after = float(data.get("parameters", {}).get("retry_after", 1))
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            retry_after = 1.0
                        log.info(
                            "Telegram rate limit (429) – waiting %.1fs before retry (attempt %d/%d)",
                            retry_after, attempt + 1, max_attempts,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status >= 500:
                        wait = 2 ** attempt
                        log.info(
                            "Telegram server error (%d) – retrying in %ds (attempt %d/%d)",
                            resp.status, wait, attempt + 1, max_attempts,
                        )
                        await asyncio.sleep(wait)
                        continue

                    return False

            except asyncio.TimeoutError:
                wait = 2 ** attempt
                log.warning(
                    "Telegram sendPhoto timeout – retrying in %ds (attempt %d/%d)",
                    wait, attempt + 1, max_attempts,
                )
                await asyncio.sleep(wait)
                continue
            except Exception as exc:
                log.error("Telegram sendPhoto error: %s", exc)
                return False

        return False

    # ------------------------------------------------------------------
    # Rich signal formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_md(text: str) -> str:
        """Escape Markdown V1 special characters in dynamic text fields.

        Telegram's legacy MarkdownV1 uses ``*``, ``_``, `` ` ``, and ``[``
        as formatting markers.  Dynamic values such as ``liquidity_info``
        may contain these characters and must be escaped so they are rendered
        as literal text rather than misinterpreted as entity boundaries.
        """
        for ch in ("\\", "*", "_", "`", "["):
            text = text.replace(ch, f"\\{ch}")
        return text

    @staticmethod
    def format_signal(sig: Signal) -> str:
        """Produce the rich, emoji-laden Telegram message for a signal."""
        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
            "360_GEM": "💎",
        }
        emoji = chan_emojis.get(sig.channel, "📡")
        dir_emoji = "🚀" if sig.direction == Direction.LONG else "⬇️"
        dir_word = sig.direction.value

        entry_text = f"🚀 Entry: `{fmt_price(sig.entry)}`"
        if sig.entry_zone:
            entry_text = (
                f"🚀 Entry: `{fmt_price(sig.entry)}`"
                f" | Zone: `{TelegramBot._escape_md(sig.entry_zone)}`"
            )

        lines = [
            f"{emoji} *{TelegramBot._escape_md(sig.channel)} ALERT* 💎",
            f"Pair: *{TelegramBot._escape_md(sig.symbol)}*",
            f"📈 *{dir_word}* {dir_emoji}",
            entry_text,
            f"🛡️ SL: `{fmt_price(sig.stop_loss)}`",
            f"🎯 TP1: `{fmt_price(sig.tp1)}`",
            f"🎯 TP2: `{fmt_price(sig.tp2)}`",
        ]
        if sig.tp3 is not None:
            lines.append(f"🎯 TP3: `{fmt_price(sig.tp3)}`")
        else:
            lines.append("🎯 TP3: Dynamic/trailing")

        if sig.trailing_active:
            lines.append(f"💹 Trailing Active ({TelegramBot._escape_md(sig.trailing_desc)})")

        lines.append(f"🤖 Confidence: *{sig.confidence:.0f}%*")
        if sig.component_scores:
            lines.append(
                "🧩 Quality: *{}* | M:{:.0f} S:{:.0f} E:{:.0f} R:{:.0f} C:{:.0f}".format(
                    TelegramBot._escape_md(sig.quality_tier),
                    sig.component_scores.get("market", 0.0),
                    sig.component_scores.get("setup", 0.0),
                    sig.component_scores.get("execution", 0.0),
                    sig.component_scores.get("risk", 0.0),
                    sig.component_scores.get("context", 0.0),
                )
            )

        sentiment_line = f"📰 AI Sentiment: {sig.ai_sentiment_label}"
        if sig.ai_sentiment_summary:
            sentiment_line += f" — {TelegramBot._escape_md(sig.ai_sentiment_summary)}"
        lines.append(sentiment_line)

        if sig.setup_class and sig.setup_class != "UNCLASSIFIED":
            setup_label = sig.setup_class.replace("_", " ").title()
            lines.append(f"🧠 Setup: {TelegramBot._escape_md(setup_label)}")
        lines.append(f"⚠️ Risk: {TelegramBot._escape_md(sig.risk_label)}")
        lines.append(f"📊 Market Phase: {TelegramBot._escape_md(sig.market_phase)}")
        lines.append(f"💧 Liquidity Pool: {TelegramBot._escape_md(sig.liquidity_info)}")
        if sig.invalidation_summary:
            lines.append(f"🧱 Invalidation: {TelegramBot._escape_md(sig.invalidation_summary)}")
        if sig.analyst_reason:
            lines.append(f"📝 Thesis: {TelegramBot._escape_md(sig.analyst_reason)}")
        if sig.execution_note:
            lines.append(f"⏱️ Execution: {TelegramBot._escape_md(sig.execution_note)}")
        lines.append(f"⏰ Time: `{fmt_ts(sig.timestamp)}`")

        return "\n".join(lines)

    @staticmethod
    def format_free_signal(sig: Signal) -> str:
        """Wrap a signal in the free-channel header/footer."""
        header = "🆓 *FREE SIGNAL OF THE DAY* 🆓\n\n"
        body = TelegramBot.format_signal(sig)
        footer = (
            "\n\n📚 _Tip: Scalping requires discipline. Always use a stop-loss"
            " and manage risk._\n📊 _Market Phase helps gauge overall conditions."
            " Premium gets all signals!_"
        )
        return header + body + footer

    @staticmethod
    def format_gem_signal(
        symbol: str,
        current_price: float,
        ath: float,
        drawdown_pct: float,
        x_potential: float,
        accumulation_days: int,
        volume_ratio: float,
        confidence: float,
        timestamp: float,
    ) -> str:
        """Format a 360_GEM macro-reversal signal for Telegram."""
        import datetime

        ts_str = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"💎 360\\_GEM ALERT — POTENTIAL x{x_potential:.0f} 🚀",
            f"Pair: `{symbol}`",
            "📈 LONG \\(Macro Reversal\\)",
            f"💰 Current Price: `{fmt_price(current_price)}`",
            f"📊 ATH: `{fmt_price(ath)}` | Drawdown: `{drawdown_pct:.0f}%`",
            f"🏗️ Accumulation Base: `{accumulation_days}` days",
            f"📈 Volume Surge: `{volume_ratio:.1f}x` average",
            "🎯 Target: Previous ATH region",
            "🛡️ SL: Below accumulation base",
            f"🤖 Confidence: `{confidence:.0f}%`",
            f"⏰ Time: `{ts_str}`",
        ]
        return "\n".join(lines)

    @staticmethod
    def format_highlight_message(sig: Signal, tp_level: int, tp_pnl_pct: float) -> str:
        """Format an eye-catching winning trade highlight for the free channel."""
        from src.utils import utcnow

        chan_emojis = {
            "360_SCALP": "⚡",
            "360_SWING": "🏛️",
            "360_SPOT": "📈",
            "360_GEM": "💎",
        }
        chan_emoji = chan_emojis.get(sig.channel, "📡")
        dir_emoji = "🚀" if sig.direction.value == "LONG" else "⬇️"
        tp_emoji = "✅" * tp_level  # ✅✅ for TP2, ✅✅✅ for TP3

        # Determine the TP price to display
        if tp_level == 3 and sig.tp3:
            tp_price = sig.tp3
        elif tp_level == 2:
            tp_price = sig.tp2
        else:
            tp_price = sig.tp1

        # Hold duration
        hold_secs = (utcnow() - sig.timestamp).total_seconds()
        if hold_secs < 60:
            duration_str = f"{hold_secs:.0f} seconds"
        elif hold_secs < 3600:
            duration_str = f"{hold_secs / 60:.0f} minutes"
        else:
            duration_str = f"{hold_secs / 3600:.1f} hours"

        lines = [
            "🏆 *WINNING TRADE HIGHLIGHT* 🏆",
            "",
            f"{chan_emoji} {TelegramBot._escape_md(sig.channel)} | "
            f"*{TelegramBot._escape_md(sig.symbol)}* *{sig.direction.value}* {dir_emoji}",
            "",
            f"📊 Result: *TP{tp_level} HIT* {tp_emoji}",
            f"💰 Entry: `{fmt_price(sig.entry)}` → TP{tp_level}: `{fmt_price(tp_price)}`",
            f"📈 PnL: *+{tp_pnl_pct:.2f}%*",
            f"⏱️ Duration: {duration_str}",
        ]

        if sig.setup_class and sig.setup_class != "UNCLASSIFIED":
            setup_label = sig.setup_class.replace("_", " ").title()
            lines.append(f"🧠 Setup: {TelegramBot._escape_md(setup_label)}")
        if sig.market_phase:
            lines.append(f"📊 Market Phase: {TelegramBot._escape_md(sig.market_phase)}")
        lines.append(f"🤖 Confidence: *{sig.confidence:.0f}%*")
        if sig.quality_tier:
            lines.append(f"🧩 Quality: *{TelegramBot._escape_md(sig.quality_tier)}*")

        lines.extend([
            "",
            "💡 _Our engine caught this move in real-time._",
            "📲 _Join Premium for ALL signals!_",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_daily_recap(summary: dict) -> str:
        """Format the daily performance recap for the free channel."""
        lines = [
            "📊 *DAILY PERFORMANCE RECAP* 📊",
            "",
        ]

        best = summary.get("best_trade")
        if best:
            lines.append(
                f"🏆 Best Trade: *{TelegramBot._escape_md(best.symbol)}* "
                f"{best.direction} *+{best.signal_quality_pnl_pct:.2f}%* "
                f"({TelegramBot._escape_md(best.channel)})"
            )

        lines.extend([
            f"⚡ Total Signals: *{summary['total']}*",
            f"✅ Wins: {summary['wins']} | ❌ Losses: {summary['losses']} | "
            f"➖ BE: {summary['breakeven']}",
            f"📈 Win Rate: *{summary['win_rate']:.0f}%*",
            f"💰 Avg PnL: *{summary['avg_pnl']:+.2f}%*",
        ])

        top_trades = summary.get("top_trades", [])
        if top_trades:
            lines.extend(["", "🔥 *Top 3 Trades:*"])
            for i, t in enumerate(top_trades, 1):
                chan_short = t.channel.replace("360_", "")
                lines.append(
                    f"{i}. *{TelegramBot._escape_md(t.symbol)}* {t.direction} "
                    f"+{t.signal_quality_pnl_pct:.2f}% ({TelegramBot._escape_md(chan_short)})"
                )

        lines.extend([
            "",
            "📲 _Premium gets ALL signals in real-time_",
            "🆓 _Free channel shows highlights only_",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Admin command polling
    # ------------------------------------------------------------------

    async def poll_commands(self, handler, on_new_member=None) -> None:
        """Long-poll for commands from any chat that starts with ``/``.

        Admin-only commands are gated inside the *handler* by comparing
        ``chat_id`` against ``TELEGRAM_ADMIN_CHAT_ID``.

        Parameters
        ----------
        handler:
            Async callable ``(text, chat_id)`` that handles ``/`` commands.
        on_new_member:
            Optional async callable ``(user_id)`` invoked when a user joins
            one of the bot's channels (``my_chat_member`` update with status
            changing from ``left``/``kicked`` to ``member``).
        """
        self._running = True
        _allowed: str = json.dumps(["message", "my_chat_member"])
        # Clear stale updates before starting to poll so commands queued
        # during a long boot (pair seeding, etc.) are not re-processed.
        try:
            session = await self._ensure_session()
            url = f"{self._base}/getUpdates"
            params: dict[str, str] = {"offset": "-1", "timeout": "0", "allowed_updates": _allowed}
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("result", [])
                    if results:
                        self._offset = results[-1]["update_id"] + 1
                        log.info("Cleared %d stale Telegram updates", len(results))
        except Exception as exc:
            log.debug("Failed to clear stale updates: %s", exc)
        while self._running:
            try:
                if not self._token:
                    await asyncio.sleep(30)
                    continue
                session = await self._ensure_session()
                url = f"{self._base}/getUpdates"
                params = {"offset": str(self._offset), "timeout": "20", "allowed_updates": _allowed}
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
                    if text.startswith("/"):
                        await handler(text, chat_id)
                    # Handle channel subscription events
                    mcm = update.get("my_chat_member", {})
                    if mcm and on_new_member is not None:
                        new_status = mcm.get("new_chat_member", {}).get("status", "")
                        old_status = mcm.get("old_chat_member", {}).get("status", "")
                        if new_status == "member" and old_status in ("left", "kicked"):
                            user_id = str(mcm.get("from", {}).get("id", ""))
                            if user_id:
                                try:
                                    await on_new_member(user_id)
                                except Exception as exc:
                                    log.debug("Welcome DM failed for user %s: %s", user_id, exc)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("Command poll error: %s", exc)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
