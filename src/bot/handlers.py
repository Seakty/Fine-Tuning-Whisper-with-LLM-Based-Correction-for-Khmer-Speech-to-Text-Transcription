"""Telegram message handlers (FR-1, FR-6, FR-7, FR-8).

Flow: voice/audio in -> validate -> download -> convert -> transcribe -> correct -> reply.

Error handling (FR-7): every failure path ends with a message to the user; nothing is
allowed to leave the chat stuck on "Processing...". Blocking work runs via
asyncio.to_thread() so one slow request doesn't freeze others (FR-8).
"""
import asyncio
import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..audio.convert import prepare_audio
from ..audio.validate import AudioError, AudioValidationError, validate_metadata
from ..asr.infer import transcribe
from ..asr.model import ASRError
from ..correction.fallback import correct_with_fallback

logger = logging.getLogger(__name__)

PIPELINE_STEP_TIMEOUT_SEC = 120   # per step (convert, transcribe)
TELEGRAM_MAX_CHARS = 4096

_MIME_TO_SUFFIX = {
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


def _guess_suffix(message, media) -> str:
    """Telegram file paths have no extension, so work it out from what Telegram tells us."""
    if message.voice:
        return ".ogg"  # voice notes are always Opus-in-Ogg

    file_name = getattr(media, "file_name", None)
    if file_name and Path(file_name).suffix:
        return Path(file_name).suffix.lower()

    mime = (getattr(media, "mime_type", None) or "").lower()
    return _MIME_TO_SUFFIX.get(mime, "")


def _split_text(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split long text into Telegram-sized chunks, preferring to cut at newlines."""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


async def _deliver(message, status_msg, text: str) -> None:
    """Send the final text (FR-6). Falls back to a fresh reply if editing fails."""
    chunks = _split_text(text)
    for i, chunk in enumerate(chunks):
        try:
            if i == 0:
                await status_msg.edit_text(chunk)
            else:
                await message.reply_text(chunk)
        except TelegramError as e:
            logger.warning("Delivery via edit/reply failed (%r); retrying as new message", e)
            try:
                await message.reply_text(chunk)
            except TelegramError:
                logger.exception("Could not deliver result to user")
                return


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a Khmer voice message and I'll transcribe it to text."
    )


async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles voice notes, audio files, and audio sent as a document (FR-1)."""
    message = update.message
    media = message.voice or message.audio or message.document
    if media is None:
        return

    # --- Layer 1: fail fast, before downloading anything -------------------------
    suffix = _guess_suffix(message, media) or ".ogg"
    try:
        validate_metadata(
            f"input{suffix}",
            getattr(media, "file_size", None),
            getattr(media, "duration", None),
        )
    except AudioValidationError as e:
        logger.info("Rejected %s: %s", media.file_unique_id, e.detail)
        await message.reply_text(e.user_message)
        return

    status_msg = await message.reply_text("Processing your voice message...")

    # --- Layer 2: download -> convert -> transcribe ------------------------------
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            raw_path = tmp_dir_path / f"input_{media.file_unique_id}{suffix}"

            try:
                tg_file = await context.bot.get_file(media.file_id)
                await tg_file.download_to_drive(custom_path=str(raw_path))
            except Exception as e:
                logger.warning("Telegram file download failed: %r", e)
                await status_msg.edit_text(
                    "I couldn't download that file. Please try sending it again."
                )
                return

            wav_path = await asyncio.wait_for(
                asyncio.to_thread(prepare_audio, raw_path, str(tmp_dir_path)),
                timeout=PIPELINE_STEP_TIMEOUT_SEC,
            )
            raw_text = await asyncio.wait_for(
                asyncio.to_thread(transcribe, wav_path),
                timeout=PIPELINE_STEP_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        logger.error("Pipeline timed out for %s", media.file_unique_id)
        await status_msg.edit_text(
            "That took too long to process. Please try a shorter clip."
        )
        return
    except (AudioError, ASRError) as e:
        detail = getattr(e, "detail", repr(e))
        logger.warning("Pipeline failed for %s: %s", media.file_unique_id, detail)
        await status_msg.edit_text(e.user_message)
        return
    except Exception:
        logger.exception("Unexpected pipeline failure for %s", media.file_unique_id)
        await status_msg.edit_text(
            "Sorry, something went wrong. Please try again in a moment."
        )
        return

    if not raw_text or not raw_text.strip():
        await status_msg.edit_text("I didn't hear anything in that clip.")
        return

    # --- Layer 3: correction (never allowed to block delivery) -------------------
    final_text = raw_text
    try:
        result = await asyncio.to_thread(correct_with_fallback, raw_text)
        final_text = result.text or raw_text
        if result.used_fallback:
            logger.info("Sent raw ASR text (correction fallback) for %s: %s",
                        media.file_unique_id, result.error_detail)
        else:
            logger.info("Gemini-corrected %s | raw: %s | corrected: %s",
                        media.file_unique_id, raw_text, final_text)
    except Exception:
        logger.exception("Correction step crashed for %s; sending raw text",
                         media.file_unique_id)

    await _deliver(message, status_msg, final_text)


async def unsupported_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all for text, photos, stickers, non-audio documents, etc."""
    await update.message.reply_text(
        "I can only transcribe Khmer voice messages or audio files. "
        "Please send one of those."
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global safety net for anything the handlers didn't anticipate."""
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Sorry, something went wrong. Please try again in a moment."
            )
        except TelegramError:
            pass