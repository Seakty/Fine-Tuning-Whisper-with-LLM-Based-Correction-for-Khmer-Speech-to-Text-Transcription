"""CLI smoke test / benchmark:
    python -m src.asr path/to/clip.wav [--adapter DIR] [--runs 3] [--fp16] [--no-merge]
"""
import argparse
import sys

from . import ASRError, benchmark, get_asr
from .model import DEFAULT_ADAPTER_DIR

p = argparse.ArgumentParser()
p.add_argument("wav")
p.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--fp16", action="store_true")
p.add_argument("--no-merge", action="store_true")
args = p.parse_args()

try:
    asr = get_asr(adapter_dir=args.adapter, fp16=args.fp16, merge=not args.no_merge)
    r = benchmark(args.wav, asr, runs=args.runs)
except ASRError as e:
    sys.exit(f"FAILED: {e.user_message}\n  detail: {e.detail}")

print(f"Transcript : {r['text']}")
print(f"Device     : {r['device']}")
print(f"Audio      : {r['audio_sec']:.2f}s")
print(f"Warm-up run: {r['warmup_sec']:.2f}s  (first call, slower - not counted)")
print(f"Timed runs : mean {r['mean_sec']:.2f}s  (min {r['min_sec']:.2f}, max {r['max_sec']:.2f})")
print(f"Real-time factor: {r['real_time_factor']:.2f}x  (below 1.0 = faster than real time)")