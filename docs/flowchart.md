[Start]
   ↓
(Literature Review — low-resource ASR)
   ↓
(Baseline Evaluation)
  ← whisper-small-khmer-v2, untouched
  ← FLEURS Khmer test set
   ↓
  ✓ Baseline CER/WER
   ↓
(Data Acquisition & Augmentation)
  ← Inherited: FLEURS + OpenSLR + km-speech-corpus
  ← + DDD-Cambodia corpus
  ← + Phone-mic augmentation (noise/bitrate simulation)
   ↓
  ✓ Expanded training dataset
   ↓
(LoRA Fine-Tuning)
  ← whisper-small-khmer-v2 + LoRA adapter
   ↓
  ✓ Fine-tuned adapter weights
   ↓
(Fine-Tuned Model Evaluation) ◇
  split into two parallel branches:
    → FLEURS held-out set
    → Real-world voice message set
  both feed into:
   ↓
  ✓ Improvement Delta (fine-tuned vs. baseline)
   ↓
(LLM Correction Layer Integration)
  ← Gemini, correction-only prompt
   ↓
(LLM-Corrected Evaluation) ◇
  same two sets (FLEURS + real-world)
   ↓
  ✓ Enhancement Delta (LLM-corrected vs. fine-tuned)
   ↓
(Full Pipeline Comparison) ◇
  Baseline vs. Fine-tuned vs. Fine-tuned+LLM
  across both eval sets
   ↓
  ✓ Consolidated results table
   ↓
(Telegram Bot Integration & Deployment)
  Audio ingestion → ASR → LLM correction → delivery
   ↓
[End]