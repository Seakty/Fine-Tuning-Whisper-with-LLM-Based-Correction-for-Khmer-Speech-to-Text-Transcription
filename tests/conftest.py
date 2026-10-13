import math
import struct
import wave
from pathlib import Path

import pytest


def _make_wav(path, seconds=1.0, rate=44100, channels=2, freq=440) -> Path:
    """Write a sine-tone PCM wav. Defaults (44.1 kHz stereo) are deliberately NOT
    16 kHz mono so conversion has real work to do."""
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        sample = int(12000 * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<h", sample) * channels
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return Path(path)


@pytest.fixture
def make_wav():
    return _make_wav