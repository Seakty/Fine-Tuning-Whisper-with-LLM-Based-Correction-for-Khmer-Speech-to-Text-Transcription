# Requirements Analysis & Project Structure

**Project:** Intelligent Khmer Voice-to-Text Telegram Bot with LLM Post-Processing
**Author:** Sambath Seakty

---

## Part 1: Requirements Analysis

### 1.1 Functional Requirements

| ID   | Requirement                  | Description                                                                                                                |
| ---- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| FR-1 | Voice message ingestion      | Bot accepts `.ogg` voice notes and forwarded audio files via Telegram                                                      |
| FR-2 | Audio preprocessing          | Convert input audio to 16kHz mono `.wav` for Whisper compatibility                                                         |
| FR-3 | Speech-to-text transcription | Fine-tuned Whisper model transcribes audio to raw Khmer text                                                               |
| FR-4 | Text post-processing         | LLM (Gemini) corrects grammar/semantic errors in raw transcription                                                         |
| FR-5 | Fallback correction          | If Gemini API is unavailable, system falls back to raw ASR output or a local correction method, not a hard failure         |
| FR-6 | Result delivery              | Final corrected text sent back to the user's Telegram chat                                                                 |
| FR-7 | Error handling               | Bot gives a clear message on unsupported file types, oversized files, or failed transcription, instead of silently failing |
| FR-8 | Concurrency support          | Bot handles multiple users sending audio at overlapping times without blocking                                             |

### 1.2 Non-Functional Requirements

| ID    | Requirement     | Target                                                                                                               |
| ----- | --------------- | -------------------------------------------------------------------------------------------------------------------- |
| NFR-1 | Latency         | End-to-end under 5 seconds for a 10-second audio clip                                                                |
| NFR-2 | Accuracy        | CER/WER measurably improved vs. the `whisper-small-khmer-v2` baseline                                                |
| NFR-3 | Availability    | Bot recoverable after a crash/restart without losing in-flight requests unnecessarily                                |
| NFR-4 | Cost            | Training stays within free/low-cost tiers (Colab free or Pro, Gemini free-tier limits respected)                     |
| NFR-5 | Reproducibility | Training pipeline and data sources documented well enough to rerun or extend later (for thesis defense and write-up) |
| NFR-6 | Maintainability | Codebase modular enough that ASR and LLM-correction stages can be swapped/upgraded independently                     |

### 1.3 Data Requirements

- Inherited training data from the `v2` checkpoint: FLEURS (Khmer) + OpenSLR SLR42 + km-speech-corpus
- Additional data (scope to be finalized — candidates: DDD-Cambodia Khmer ASR Cultural Dataset, manually recorded samples)
- A held-out **evaluation set** kept separate from all training data, ideally including some spontaneous/phone-mic-quality audio (not only studio-quality FLEURS-style recordings) so reported metrics reflect real Telegram usage
- All datasets tracked with source, license, and hour count in one place (avoids losing track of what's allowed to be published in the thesis)

### 1.4 System / Environment Requirements

- **Training:** Google Colab (T4 GPU, free tier) with Google Drive checkpointing; Kaggle Notebooks as backup compute
- **Inference/hosting:** Hugging Face Spaces or Render
- **Languages/frameworks:** Python, Hugging Face `transformers` + `peft` (LoRA), `python-telegram-bot`, `librosa`/`ffmpeg`
- **External APIs:** Google Gemini API (post-processing), subject to free-tier rate limits — factor this into concurrency testing (FR-8)

### 1.5 Constraints

- Limited weekly time available (internship, teaching work, and two other concurrent projects) — timeline should assume Colab-only sessions in blocks rather than long continuous work
- Free-tier compute session limits (Colab ~12h cap, disconnects after idle) shape how training must be checkpointed
- Khmer is a low-resource language — data scarcity is a structural constraint, not something solvable purely with more engineering time

### 1.6 Success Criteria (Definition of Done)

- [ ] Fine-tuned model outperforms the `v2` baseline on CER, on a held-out test set
- [ ] LLM post-processing measurably reduces error rate vs. raw ASR output (LLM Enhancement Delta documented)
- [ ] Telegram bot deployed and functional end-to-end (voice in → corrected text out)
- [ ] Latency target met for a standard 10s clip
- [ ] All data sources and licenses documented for the thesis write-up

---
