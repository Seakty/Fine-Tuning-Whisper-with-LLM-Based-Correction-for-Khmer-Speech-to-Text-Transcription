# Evaluation Results — Baseline vs. Fine-Tuned

**Base checkpoint:** `seanghay/whisper-small-khmer-v2`
**Fine-tuning:** LoRA (r=16, alpha=32, target_modules=[q_proj, v_proj]) on FLEURS + OpenSLR + km-speech-corpus (inherited) + DDD-Cambodia + augmented data, 12,762 steps / 3 epochs
**Eval sets:** FLEURS Khmer test split (776 samples, studio-quality read speech) and a
real-world holdout set (38 samples, casual phone-mic Telegram-style voice notes, self-collected
per `DATA-06`) — both held out, never trained on
**Date:** [today's date]

## Headline Metrics

| Metric               | Baseline   | Fine-Tuned | Improvement                                                       |
| -------------------- | ---------- | ---------- | ----------------------------------------------------------------- |
| **CER (FLEURS)**     | **41.10%** | **32.27%** | **8.83 pp (21.5% relative)**                                      |
| WER (FLEURS)         | 457.15%    | 108.11%    | Large drop, but see caveat below — not used as the primary signal |
| **CER (Real-World)** | **57.89%** | **50.47%** | **7.42 pp (12.8% relative)**                                      |

CER is the primary metric for this project. The fine-tuned model reduces character-level
errors by roughly a fifth relative to the untouched `v2` checkpoint on the held-out FLEURS
test set, and by about an eighth on the real-world evaluation set.

### FLEURS vs. Real-World gap

Both models perform noticeably worse on the real-world set (38 casual, phone-mic Telegram-style
voice notes, `DATA-06`) than on FLEURS (776 studio-quality read-speech samples) — roughly
+16.8 pp CER for the baseline and +18.2 pp for the fine-tuned model. This is expected and
important: FLEURS numbers alone would overstate how the bot performs on its actual real-world
use case. The relative improvement from fine-tuning holds up in both conditions (21.5%
relative on FLEURS, 12.8% relative on real-world), though the fine-tuning gain is somewhat
smaller on harder, noisier audio — likely because the DDD-Cambodia + augmentation data still
doesn't fully capture the acoustic variety of genuine phone-mic recordings. This gap is a
legitimate, worth-reporting limitation rather than something to explain away, and it directly
motivates keeping both eval sets going forward (`EVAL-06`) rather than reporting FLEURS alone.

## Why WER is not trustworthy here

Khmer script does not use consistent word spacing, and `jiwer`'s WER calculation
splits on whitespace. The baseline model in particular inserts far more spaces than
the reference transcript (e.g. a 3-word reference producing a 20-word prediction
by space count), which inflates WER far beyond what it reflects for actual
transcription quality. CER is used as the primary metric for this reason, per
the original project scope. The fine-tuned model's large WER drop is a real and
consistent signal (see qualitative examples below) — it produces noticeably
fewer, more reference-like word-boundary breaks than the baseline — but the
absolute WER number itself should still be treated as directional, not exact.

## Observed Error Patterns (qualitative)

**Baseline:**

1. **Word-level hallucination:** the model occasionally substitutes plausible-sounding
   but incorrect Khmer words (e.g. reference "គាត់និយាយត្រូវ" → predicted "កុលត្យា ត្រូវ").
2. **Code-switched terms mishandled:** English loanwords embedded in Khmer speech
   (e.g. "nyiragongo", "mountain gorilla") are transcribed as garbled Khmer-sounding
   syllables rather than recognized or transliterated sensibly.
3. **Over-segmentation:** output is broken into far more space-separated fragments
   than the reference, syllable-by-syllable in places (e.g. "ការដំណាច អង្គការ ប្រជាជន
   ហ្រស្វិច ខាង លិច ឈាម ខាងត្បូង...").

**Fine-tuned:**

- Retains some of the same substitution-style errors as the baseline, but produces
  noticeably fewer, more natural word-length spacing breaks — closer to the reference's
  segmentation style (e.g. "អង្គការប្រជាជន៍អស្រ្វិខាងលិច ឈាមខាងត្បូង ស្វាប់ពរ..." vs. the
  baseline's much more fragmented equivalent on the same file).
- Code-switched terms (e.g. "faa", "nick jayne") are still imperfectly handled but no
  longer fully absent/garbled the way some baseline outputs were.

These patterns still motivate the hybrid architecture: an LLM correction layer is well
positioned to catch and fix the remaining local word-level errors, which is part of the
core thesis contribution.

## Evaluation Bug Postmortem (for the record / Discussion chapter)

The first fine-tuning evaluation attempt reported CER unchanged at 41.08% (vs. 41.10%
baseline) despite a healthy, steadily-decreasing training loss curve over 12,762 steps.
Root cause, found through step-by-step isolation:

1. **The training run itself was legitimate.** Loss dropped smoothly from ~1.17 to ~0.11
   with no anomalies across two manual interrupt/resume cycles.
2. **The trained adapter was overwritten with an untrained one at save time.** After a
   Colab disconnect right at the end of the finished training run, the notebook was
   restarted and the model-setup cell (`get_peft_model(...)`) was re-run to restore a
   working session — but the training-resume cell was skipped, since training already
   looked complete. `trainer.save_model(FINAL_ADAPTER_DIR)` then saved this fresh,
   never-trained LoRA wrapper (zero-initialized `lora_B` weights) instead of the real
   trained checkpoint, with no error or warning.
3. **A resume-safe prediction cache masked the problem for two more evaluation runs**,
   since a prior partially-completed prediction file caused the eval notebook to skip
   re-transcription entirely and simply rescore old cached (effectively baseline-identical)
   predictions.
4. **Fix:** verified the real trained weights were intact and untouched in
   `checkpoints/lora_run1/checkpoint-12762` (`lora_B` norm 85.83, vs. 0.0 in the broken
   `final_adapter`), promoted that checkpoint to `final_adapter/`, re-verified the adapter
   changed model output via a direct fine-tuned-vs-plain-base generation comparison, then
   cleared the stale prediction cache and re-ran the full evaluation.

Practical takeaway for future runs: always assert `trainer.state.global_step` (or check a
`lora_B` weight norm) immediately before and after any `save_model()` call, especially
after a session restart, and treat a fine-tuned eval result that's suspiciously identical
to baseline as a measurement red flag rather than a modeling conclusion.

## Next steps

- ~~Fine-tune on expanded dataset (DDD-Cambodia + augmentation)~~ ✅ Done
- ~~Re-run this same eval script on the fine-tuned model for the Improvement Delta~~ ✅ Done — 8.83 pp / 21.5% relative CER improvement (FLEURS)
- ~~evaluate on real-world holdout audio (`DATA-06`)~~ ✅ Done — 7.42 pp / 12.8% relative CER improvement (real-world)
- Run LLM post-processing on the fine-tuned model's output to get the LLM Enhancement Delta (`EVAL-05`) — ideally measured on both eval sets
- Full pipeline comparison: baseline vs. fine-tuned vs. fine-tuned+LLM (`EVAL-06`)
