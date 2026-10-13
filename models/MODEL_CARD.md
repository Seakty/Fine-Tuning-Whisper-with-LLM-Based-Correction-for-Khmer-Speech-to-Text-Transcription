# Model Card — Khmer Whisper LoRA Adapter (`final_adapter`)

## Model Details

- **Base model:** [`seanghay/whisper-small-khmer-v2`](https://huggingface.co/seanghay/whisper-small-khmer-v2)
- **Adapter type:** LoRA (Low-Rank Adaptation) via PEFT — base model weights frozen, only the adapter is trained
- **LoRA config:** `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj"]`, `lora_dropout=0.05`, `bias="none"`
- **Trainable parameters:** 1,769,472 out of 243,504,384 total (0.73%)
- **Training length:** 12,762 steps, 3 epochs
- **Author:** Sambath Seakty
- **Part of:** Intelligent Khmer Voice-to-Text Telegram Bot with LLM Post-Processing (thesis project)

## Intended Use

Acoustic transcription stage of a hybrid Khmer speech-to-text pipeline: converts raw
16kHz Khmer audio (Telegram voice notes) into raw Khmer text, which is then passed to
an LLM (Gemini) correction layer before being returned to the user. **Not intended to
be used standalone as a final-answer transcription system** — the pipeline's accuracy
claims depend on the downstream correction step, which has not yet been evaluated
(see Limitations).

## Training Data

| Source                                  | Status                         | Notes                                                                                                                                                                        |
| --------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FLEURS (Khmer)                          | Inherited from `v2` checkpoint | Studio-quality, read speech                                                                                                                                                  |
| OpenSLR SLR42                           | Inherited from `v2` checkpoint | —                                                                                                                                                                            |
| km-speech-corpus                        | Inherited from `v2` checkpoint | —                                                                                                                                                                            |
| DDD-Cambodia Khmer ASR Cultural Dataset | Added this project             | CC-BY-SA licensed — attribution required in any redistribution                                                                                                               |
| Augmented copies of the above           | Added this project             | Synthetic phone-mic-quality variants (background noise, static, bitrate/gain reduction) via `audiomentations`, generated to better match real Telegram voice-note conditions |

Total fine-tuning set: 68,059 samples. Full source/license/hour breakdown tracked in
`data/DATASETS.md`.

**Held-out evaluation set:** FLEURS Khmer test split, 776 samples — kept fully separate
from all training data at every stage. A real-world spontaneous/phone-mic evaluation set
(`DATA-06`) is planned but not yet collected; all metrics below are FLEURS-only and have
not yet been validated against casual, real Telegram-style audio.

## Evaluation

| Metric                           | Baseline (`v2`, untouched) | This adapter | Improvement                               |
| -------------------------------- | -------------------------- | ------------ | ----------------------------------------- |
| **CER (FLEURS)**                 | 41.10%                     | **32.27%**   | 8.83 pp (21.5% relative)                  |
| WER (FLEURS)                     | 457.15%                    | 108.11%      | Large drop, directional only — see caveat |
| **CER (Real-World, 38 samples)** | 57.89%                     | **50.47%**   | 7.42 pp (12.8% relative)                  |

CER is the primary metric for this project. WER is reported for comparability with prior
Khmer ASR work but is not fully trustworthy for either model: Khmer script has no
consistent word spacing, and `jiwer`'s whitespace-based WER calculation inflates error
counts on both models, more severely on the baseline's heavily over-segmented output.
The real-world set is a small (38-sample) self-collected holdout of casual, phone-mic
Telegram-style voice notes — both models perform meaningfully worse here than on FLEURS,
which is expected and reflects the actual deployment condition more closely than
studio-quality read speech does. Full methodology and qualitative examples in
`docs/baseline_results.md`.

## Known Limitations

- **Code-switched terms:** English loanwords embedded in Khmer speech (e.g. "faa",
  "nick jayne") are still imperfectly transcribed — improved over baseline but not
  reliably recognized or transliterated.
- **Residual word-level substitution errors:** the model still occasionally produces
  plausible-sounding but incorrect Khmer words, a pattern the downstream LLM correction
  layer is intended to catch (not yet evaluated — `EVAL-05`).
- **Performance drops noticeably on real-world audio.** CER rises from 32.27% (FLEURS,
  studio-quality) to 50.47% on a 38-sample real-world holdout of casual, phone-mic
  Telegram-style voice notes — the actual deployment condition. The relative gain from
  fine-tuning also shrinks somewhat on this harder set (12.8% vs. 21.5% relative on
  FLEURS), suggesting the training data still doesn't fully capture real phone-mic
  acoustic variety. Treat the FLEURS number as a best case, not a representative estimate
  of real-world accuracy. The real-world sample size (38) is also small enough that this
  number should be treated as indicative rather than statistically precise.
- **Narrow adapter scope:** LoRA was applied only to attention `q_proj`/`v_proj`
  projections at rank 16 — a relatively lightweight intervention. Larger gains may be
  possible with a higher rank, additional target modules, or more training data, at the
  cost of more compute.
- **WER should not be quoted as a standalone accuracy figure** for either model, for the
  reasons above — always pair it with CER and the spacing caveat.
- **License obligations:** this adapter's training data includes the CC-BY-SA-licensed
  DDD-Cambodia dataset. Any redistribution of this adapter or derivative work should
  carry the corresponding attribution.

## Known Issue History (for reproducibility)

An earlier evaluation of this adapter incorrectly reported no improvement over baseline
(CER 41.08% vs. 41.10%). Root cause: a Colab session restart after training completed
led to the training-resume step being skipped before saving, so an untrained,
freshly-initialized LoRA adapter was saved to disk instead of the actual trained
weights — silently, with no error. The metrics in this card reflect the corrected
adapter, verified by (1) confirming non-zero `lora_B` weight norms matching the real
training checkpoint and (2) confirming the adapter's generation output differs from the
plain base model's on held-out samples. Full postmortem in `docs/baseline_results.md`.
