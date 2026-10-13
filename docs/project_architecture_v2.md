# Project Overview & System Architecture (v2)

**Project Title:** Intelligent Khmer Voice-to-Text Telegram Bot with LLM Post-Processing
**Author:** Sambath Seakty
**Domain:** Data Science / Natural Language Processing (NLP) / MLOps
**Status:** Revised plan — starting from an existing fine-tuned checkpoint, scope for additional data still open

---

## 1. Executive Summary

This project builds an automated Khmer Speech-to-Text (STT) Telegram bot for a low-resource language with limited existing tooling. The system uses a hybrid pipeline: a fine-tuned Whisper ASR model transcribes raw audio, and an LLM correction layer cleans up grammatical and semantic errors in the output. Rather than fine-tuning from the base multilingual Whisper checkpoint, this version starts from an existing Khmer-tuned checkpoint (`seanghay/whisper-small-khmer-v2`) and extends it with additional training data and evaluation depth — positioning the contribution as "how much further can we push an already-decent Khmer ASR baseline" rather than starting from zero.

---

## 2. What Already Exists (Baseline / Prior Work)

Before this project, published Khmer Whisper fine-tunes already exist:

| Model                             | Training data                               | WER   |
| --------------------------------- | ------------------------------------------- | ----- |
| `seanghay/whisper-small-khmer`    | FLEURS + OpenSLR (SLR42)                    | ~0.65 |
| `seanghay/whisper-small-khmer-v2` | FLEURS + OpenSLR (SLR42) + km-speech-corpus | ~0.62 |

**Takeaway:** Khmer ASR is proven feasible but still has a high error rate — there is real, defensible room for improvement. This is the gap this thesis targets.

---

## 3. System Architecture Pipeline

### Stage 1: Data Ingestion (Telegram Interface)

- User sends an `.ogg` voice note or forwards an audio file to the Telegram Bot.
- Backend downloads the file and converts `.ogg` → `.wav`, normalized to 16kHz (Whisper's required input format).

### Stage 2: Hybrid AI Processing Engine

1. **Acoustic Engine (Fine-tuned Whisper)**
   - Starting checkpoint: `seanghay/whisper-small-khmer-v2` (already trained on FLEURS + OpenSLR SLR42 + km-speech-corpus).
   - Further fine-tuned on additional data (scope TBD — see Section 4).
   - Fine-tuning technique: PEFT / LoRA (Low-Rank Adaptation), same efficient approach as before, now applied on top of an already-tuned model instead of the raw multilingual base.
   - Output: raw Khmer text, still imperfect.

2. **Semantic Engine (LLM Post-Processing)**
   - Raw text passed to an LLM (Gemini API) via a strict system prompt.
   - Role: ASR error correction — fixes grammar, context mismatches, and spelling without altering meaning.
   - **Backup plan:** Gemini API is a live-demo dependency risk (rate limits / outages). Maintain a fallback path — either a cached demo example or a smaller local/open-source LLM for correction — so a defense demo isn't blocked by an API issue.

### Stage 3: Output & Delivery

- Bot sends the polished text back to the user's Telegram `chat_id`.

---

## 4. Data Scope (Open — To Be Finalized)

Starting point already covers FLEURS + OpenSLR SLR42 + km-speech-corpus (inherited from the v2 checkpoint). Additional data to layer on top — not yet decided, options to evaluate:

- **More open-source Khmer speech data** — e.g. the DDD-Cambodia Khmer ASR Cultural Dataset (larger community corpus, CC-BY-SA licensed) as a candidate to substantially increase training hours.
- **Manually collected/recorded data** — narrower in volume but can target the bot's actual real-world use case (casual, phone-mic, spontaneous speech) rather than only studio-quality read speech, which is what FLEURS provides.
- Decision on exact scope to be finalized before Phase 2 (see below) — likely driven by Colab compute budget and how much labeled data is realistically obtainable in the timeline.

---

## 5. Technology Stack

- **ASR Base Model:** `seanghay/whisper-small-khmer-v2` (starting checkpoint, not raw `whisper-small`)
- **Fine-Tuning Technique:** PEFT (LoRA)
- **Training Data:** Inherited (FLEURS + OpenSLR SLR42 + km-speech-corpus) + additional data, scope TBD
- **Post-Processing / LLM:** Google Gemini API (via Google AI Studio), with a backup correction path
- **Backend Framework:** Python (`python-telegram-bot`, `librosa`/`ffmpeg` for audio processing)
- **Training Environment:** Google Colab (T4 GPU, free tier) — see Section 6
- **Deployment & Hosting:** Hugging Face Spaces or Render

---

## 6. Training Workflow on Google Colab

Free-tier Colab (T4, 16GB VRAM) is sufficient for `whisper-small` + LoRA fine-tuning, but session behavior needs to be planned around:

- **Checkpoint to Google Drive regularly** (e.g. every N training steps) — free Colab disconnects after ~90 min idle and has a ~12h hard session cap, so an unsaved multi-hour run can be lost.
- **Resume from last checkpoint** if a session drops, rather than restarting training.
- **Fallback compute:** Kaggle Notebooks (30 GPU hours/week, free) as a backup if Colab's free quota is exhausted mid-project.
- **Colab Pro (~$10/mo)** as an option if free-tier session limits become a recurring blocker.

---

## 7. Implementation Phases

- **Phase 1: Baseline & Data Scoping**
  - Benchmark `whisper-small-khmer-v2` as-is on a held-out test set (establish the starting WER/CER).
  - Finalize additional data scope (Section 4).
- **Phase 2: Extended Fine-Tuning**
  - Apply LoRA to further fine-tune the v2 checkpoint on the expanded dataset, on Colab (with Drive checkpointing).
  - Export trained weights.
- **Phase 3: Pipeline Integration**
  - Build the Python backend linking audio conversion → Whisper inference → Gemini correction.
  - Implement the backup correction path.
- **Phase 4: Bot Deployment & Optimization**
  - Deploy to Hugging Face Spaces or Render.
  - Test concurrency (multiple simultaneous users).

---

## 8. Evaluation & Success Metrics

1. **CER (Character Error Rate)** — primary metric. Khmer script has no spaces between words, making word-level WER boundaries ambiguous; CER is the more meaningful signal for this language.
2. **WER** — reported alongside CER for comparability with prior Khmer ASR work (Section 2 table).
3. **Improvement Delta** — this project's fine-tuned model vs. the `v2` starting checkpoint, on the same held-out test set.
4. **LLM Enhancement Delta** — raw fine-tuned Whisper output vs. output after Gemini post-processing, to quantify the value the correction layer adds.
5. **System Latency** — end-to-end audio-submission-to-text-delivery time (target: < 5 seconds for a 10-second clip).
6. **Real-world robustness (secondary)** — test set should include some spontaneous, phone-mic-quality audio, not only studio-quality FLEURS-style recordings, since that's closer to actual Telegram voice notes.

---

## 9. Open Questions (to resolve before/with advisor)

- Final scope of additional training data (Section 4) — volume and source(s) still undecided.
- Whether to collect any manual/self-recorded Khmer audio, and if so, how much is realistically feasible in the timeline.
- Whether Gemini API is the final choice for post-processing or whether an open-source LLM alternative should be benchmarked too.
