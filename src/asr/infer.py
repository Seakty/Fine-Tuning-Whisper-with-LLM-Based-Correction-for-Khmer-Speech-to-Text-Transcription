"""Transcribe a 16 kHz mono .wav (from src.audio.convert) into Khmer text.

Whisper only sees a 30-second window: longer audio is silently TRUNCATED by the feature
extractor. So long clips are split into chunks (cut at the quietest point near the end of
each chunk, so words aren't sliced in half), transcribed separately, and joined.
"""
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .model import ASRError, ASRModel, get_asr

SAMPLE_RATE = 16000
MAX_CHUNK_SEC = 28.0        # under Whisper's 30 s window, with a safety margin
SEARCH_WINDOW_SEC = 6.0     # look for a pause in the last 6 s of each chunk
MIN_TAIL_SEC = 1.0          # a leftover shorter than this is merged into the previous chunk
FRAME_SEC = 0.02
SILENCE_PEAK = 1e-3         # whole clip quieter than this -> treat as silence, skip the model
MAX_TARGET_TOKENS = 448     # Whisper decoder limit


def split_audio(audio: np.ndarray, sr: int = SAMPLE_RATE, max_sec: float = MAX_CHUNK_SEC,
                search_sec: float = SEARCH_WINDOW_SEC, min_tail_sec: float = MIN_TAIL_SEC
                ) -> List[np.ndarray]:
    """Split into chunks <= max_sec, cutting at the lowest-energy frame near each boundary."""
    max_len = int(max_sec * sr)
    if len(audio) <= max_len:
        return [audio]

    frame = int(FRAME_SEC * sr)
    search_len = int(search_sec * sr)
    chunks, start = [], 0

    while len(audio) - start > max_len:
        win_start, win_end = start + max_len - search_len, start + max_len
        window = audio[win_start:win_end]
        n = len(window) // frame
        energy = (window[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
        cut = win_start + int(np.argmin(energy)) * frame + frame // 2
        chunks.append(audio[start:cut])
        start = cut

    tail = audio[start:]
    if chunks and len(tail) < int(min_tail_sec * sr):
        chunks[-1] = np.concatenate([chunks[-1], tail])   # still < 30 s
    else:
        chunks.append(tail)
    return chunks


def max_new_tokens_for(duration_sec: float, prompt_length: int) -> int:
    """Same token budget as the evaluation notebook, so behaviour matches the measured CER."""
    return min(int(duration_sec * 40) + 80, MAX_TARGET_TOKENS - prompt_length - 10)


def _load_audio(wav_path) -> np.ndarray:
    import librosa
    try:
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise ASRError("I couldn't read that audio. Please try again.",
                       detail=f"load failed: {e!r}")
    return audio


def _transcribe_chunk(chunk: np.ndarray, asr: ASRModel) -> str:
    import torch

    features = asr.processor(chunk, sampling_rate=SAMPLE_RATE,
                             return_tensors="pt").input_features.to(asr.device)
    if asr.fp16:
        features = features.half()
    n_tokens = max_new_tokens_for(len(chunk) / SAMPLE_RATE, asr.prompt_length)

    try:
        with asr.lock, torch.inference_mode():
            ids = asr.model.generate(features, max_new_tokens=n_tokens)
    except RuntimeError as e:   # includes CUDA out-of-memory
        raise ASRError("Sorry, transcription failed. Please try again in a moment.",
                       detail=f"generate failed: {e!r}")
    return asr.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe(wav_path, asr: Optional[ASRModel] = None) -> str:
    """Return the raw Khmer transcript ('' for silence). Raises ASRError on failure."""
    asr = asr or get_asr()
    audio = _load_audio(wav_path)

    if audio.size == 0:
        raise ASRError("That audio is empty.", detail="0 samples after loading")
    if float(np.abs(audio).max()) < SILENCE_PEAK:
        return ""   # Whisper tends to hallucinate text on pure silence

    pieces = [_transcribe_chunk(c, asr) for c in split_audio(audio)]
    return " ".join(p for p in pieces if p)


def benchmark(wav_path, asr: Optional[ASRModel] = None, runs: int = 3) -> dict:
    """Time transcribe(). One untimed warm-up run first (the first GPU call is always slow)."""
    asr = asr or get_asr()
    audio_sec = len(_load_audio(wav_path)) / SAMPLE_RATE

    t0 = time.perf_counter()
    text = transcribe(wav_path, asr)
    warmup = time.perf_counter() - t0

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        transcribe(wav_path, asr)
        times.append(time.perf_counter() - t0)

    mean = sum(times) / len(times)
    return {"text": text, "audio_sec": audio_sec, "warmup_sec": warmup,
            "mean_sec": mean, "min_sec": min(times), "max_sec": max(times),
            "real_time_factor": mean / audio_sec if audio_sec else float("nan"),
            "device": asr.device}