"""CLI smoke test:  python -m src.audio path/to/voice.ogg"""
import sys

from . import AudioError, TARGET_SAMPLE_RATE, prepare_audio, wav_duration_seconds

if len(sys.argv) != 2:
    sys.exit("usage: python -m src.audio <audio_file>")
try:
    out = prepare_audio(sys.argv[1])
except AudioError as e:
    sys.exit(f"FAILED: {e.user_message}\n  detail: {e.detail}")
print(f"OK -> {out}  ({wav_duration_seconds(out):.2f}s, {TARGET_SAMPLE_RATE} Hz mono)")