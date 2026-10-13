# Project Overview & System Architecture

**Project Title:** Intelligent Khmer Voice-to-Text Telegram Bot with LLM Post-Processing  
**Author:** Sambath Seakty  
**Domain:** Data Science / Natural Language Processing (NLP) / MLOps

---

## 1. Executive Summary

This project aims to solve the accessibility gap for low-resource languages in speech recognition by building an automated Khmer Speech-to-Text (STT) Telegram bot. The system utilizes a hybrid AI pipeline: a fine-tuned Automatic Speech Recognition (ASR) model transcribes the raw audio, and a Large Language Model (LLM) corrects grammatical and semantic errors in the output. The final product is deployed as a user-friendly Telegram bot, allowing users to forward or record voice messages and receive highly accurate Khmer text in real-time.

---

## 2. System Architecture Pipeline

The architecture is divided into three distinct stages: Input, Processing, and Output.

### Stage 1: Data Ingestion (Telegram Interface)

- **User Action:** User sends an `.ogg` voice note or forwards an audio file to the Telegram Bot.
- **Backend Handling:** The Python bot script receives the webhook/poll, downloads the file, and converts the `.ogg` audio to a `.wav` format normalized at a 16kHz sample rate (the required input format for Whisper).

### Stage 2: Hybrid AI Processing Engine

This is the core Data Science component, split into two sequential sub-systems:

1.  **Acoustic Engine (OpenAI Whisper):**
    - The `wav` file is processed by a fine-tuned `whisper-small` or `whisper-base` model.
    - _Output:_ Raw, unpolished Khmer text (susceptible to minor acoustic misinterpretations).
2.  **Semantic Engine (Gemini API):**
    - The raw text is passed to the Gemini API via a strict system prompt.
    - _Role:_ Acts as an ASR Error Correction layer. It fixes grammatical errors, context mismatches, and spelling mistakes without altering the original meaning.
    - _Output:_ Clean, contextually accurate Khmer text.

### Stage 3: Output & Delivery

- **Return Pipeline:** The Python script receives the polished text from Gemini.
- **User Delivery:** The bot sends the final text back to the specific user's `chat_id` in Telegram as a direct message reply.

---

## 3. Technology Stack

- **ASR Base Model:** `openai/whisper-small` (Hugging Face)
- **Fine-Tuning Technique:** PEFT (Parameter-Efficient Fine-Tuning) via LoRA (Low-Rank Adaptation)
- **Training Dataset:** Google FLEURS (Khmer subset)
- **Post-Processing / LLM:** Google Gemini API (via Google AI Studio)
- **Backend Framework:** Python (`python-telegram-bot`, `librosa` or `ffmpeg` for audio processing)
- **Deployment & Hosting:** Hugging Face Spaces (GPU/CPU hosting) or Render

---

## 4. Implementation Phases

- **Phase 1: Data Preparation & Baseline Testing**
  - Acquire the FLEURS dataset.
  - Test baseline Whisper on Khmer audio to establish a starting error rate.
- **Phase 2: Model Fine-Tuning**
  - Apply LoRA to fine-tune the Whisper model on Kaggle/Colab GPUs.
  - Export the trained weights.
- **Phase 3: Pipeline Integration**
  - Write the Python backend to link the audio conversion, Whisper inference, and Gemini API correction.
- **Phase 4: Bot Deployment & Optimization**
  - Deploy the script to a cloud server.
  - Test concurrency (multiple users sending audio at once).

---

## 5. Evaluation & Success Metrics

The academic success of the AI pipeline will be measured using:

1.  **Word Error Rate (WER) / Character Error Rate (CER):** Comparing the baseline Whisper model vs. the Fine-tuned Whisper model.
2.  **LLM Enhancement Delta:** Comparing the WER of the raw fine-tuned output vs. the final output after Gemini post-processing to quantify the value of the LLM layer.
3.  **System Latency:** Measuring the end-to-end processing time from audio submission to text delivery (target: < 5 seconds for a 10-second audio clip).
