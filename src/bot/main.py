"""Bot entrypoint. Loads the Telegram token, registers handlers, starts polling.

Run with: python -m src.bot.main
Requires TELEGRAM_BOT_TOKEN set (see .env.example).
"""
import logging
import os

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from .handlers import (
    start, voice_message_handler, unsupported_message_handler, error_handler,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _get_bot_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. See .env.example for required variables."
        )
    return token


def build_application() -> Application:
    app = Application.builder().token(_get_bot_token()).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        voice_message_handler,
    ))
    # Must come last: catches everything the handlers above didn't
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & ~filters.COMMAND,
        unsupported_message_handler,
    ))
    app.add_error_handler(error_handler)

    return app

def main() -> None:
    app = build_application()
    logger.info("Bot starting (polling)...")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()