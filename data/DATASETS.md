# Dataset Sources & Licenses — Khmer Voice-to-Text Project

| Dataset                                                        | Source                                                | License                                                        | Hours                         | Status                                          |
| -------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- | ----------------------------- | ----------------------------------------------- |
| Google FLEURS (Khmer, `km_kh`)                                 | huggingface.co/datasets/google/fleurs                 | CC-BY 4.0                                                      | ~12 hrs (train+dev+test)      | Already in baseline (`v2` checkpoint)           |
| OpenSLR SLR42                                                  | openslr.org/42                                        | CC BY-SA 4.0                                                   | 3.97 hrs (male speakers only) | Already in baseline (`v2` checkpoint)           |
| km-speech-corpus                                               | huggingface.co/datasets/seanghay/km-speech-corpus     | CC-BY 4.0                                                      | 10.4 hrs (14,943 clips)       | Already in baseline (`v2` checkpoint)           |
| DDD-Cambodia Khmer ASR Cultural Dataset (`khm-asr-cultural`)   | huggingface.co/datasets/DDD-Cambodia/khm-asr-cultural | CC-BY-SA-4.0 — **attribution to Digital Divide Data required** | 134.6 hrs (current version)   | **Acquired — in `data/raw/ddd_cambodia`**       |
| Real-world holdout (self-collected Telegram-style voice notes) | Self-collected                                        | N/A, not published                                             | ~15–20 samples (target)       | Candidate — evaluation only, never for training |

## Final Training Composition (post-merge, DATA-09)

- DDD-Cambodia (raw): 56,716 files (~134.60 hours)
- DDD-Cambodia (augmented, phone-mic simulation): 11,343 files (~20% of raw, randomly sampled)
- **Total new training data added on top of inherited `v2` checkpoint data: 68,059 files**
- Manifest: `data/processed/training_manifest.tsv` (deduplicated, shuffled, seed=42)
- No duplicates found during merge (68,059 entries before and after dedup check)

**Data quality note:** 344 of 56,716 DDD-Cambodia raw transcripts (~0.6%) contain
embedded newline characters (from multi-line recipe descriptions), causing truncated
text after the line break during metadata parsing. All corresponding audio files
remain in the training set with their (shortened) transcript intact.

**Data quality note:** 603 of 68,059 training transcripts (~0.89%) exceed Whisper's
448-token limit, likely from lengthy recipe/cultural descriptions in the DDD-Cambodia
dataset. These are truncated to 448 tokens during tokenization rather than excluded —
audio and the majority of each transcript remain usable for training.
w