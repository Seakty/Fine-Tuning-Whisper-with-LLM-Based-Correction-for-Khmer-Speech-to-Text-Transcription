# Trello Board — Khmer Voice-to-Text Telegram Bot

### 3 Lists: Done → In Progress → To Do. Each card has an ID, Domain, Description, and Sub-Tasks checklist.

---

## List: Done

**Card 1: `[SETUP-01] Scaffold Project Repository Structure`**

- Domain: Setup
- Description: Create the full project folder structure and placeholder files.
- Sub-Tasks:
  - [x] Create folders: `data/raw`, `data/processed`, `data/eval`, `notebooks/`, `src/audio`, `src/asr`, `src/correction`, `src/bot`, `src/utils`, `models/checkpoints`, `tests/`, `deployment/`, `docs/thesis_notes`
  - [x] Create placeholder files (`README.md`, `requirements.txt`, `.env.example`, `.gitignore`, notebook files, `.py` module files)
  - [x] Verify structure with `tree /F`

**Card 2: `[SETUP-02] Organize Planning Docs`**

- Domain: Setup
- Description: Move existing planning documents into `docs/`.
- Sub-Tasks:
  - [x] Move `project_architecture_v2.md` into `docs/`
  - [x] Move `requirements_analysis.md` into `docs/`
  - [x] Move `project_structure.md` into `docs/`

---

## List: In Progress

_(empty — drag the next card here when you start it)_

---

## List: To Do

**Card 3: `[DATA-01] Set Up Colab Baseline Environment`**

- Domain: Data
- Description: Prepare the Colab notebook and environment for baseline evaluation.
- Sub-Tasks:
  - [x] Install dependencies: `transformers`, `peft`, `librosa`, `jiwer`, `soundfile`
  - [x] Confirm GPU runtime is set to T4
  - [x] Mount Google Drive for checkpoint/result storage

**Card 4: `[DATA-02] Load & Run Baseline Model`**

- Domain: Data
- Description: Load the existing fine-tuned checkpoint and confirm it runs inference correctly, with no training yet.
- Sub-Tasks:
  - [x] Load `seanghay/whisper-small-khmer-v2` from Hugging Face
  - [x] Run a test transcription on 2–3 sample audio files
  - [x] Confirm output format and sample rate handling (16kHz)

**Card 5: `[DATA-03] Build Held-Out Evaluation Set`**

- Domain: Data
- Description: Assemble an evaluation set kept fully separate from any training data.
- Sub-Tasks:
  - [x] Download the FLEURS Khmer test split
  - [x] Store in `data/eval/`
  - [x] Note: real-world spontaneous audio to be added later (see Card 10)

**Card 6: `[EVAL-01] Write CER/WER Metrics Utility`**

- Domain: Evaluation
- Description: Implement reusable scoring functions for the whole project.
- Sub-Tasks:
  - [x] Implement `src/utils/metrics.py` with `compute_cer()` and `compute_wer()`
  - [x] Use `jiwer` (or equivalent) under the hood
  - [x] Add a quick unit test with a known input/output pair

**Card 7: `[EVAL-02] Run Baseline Evaluation`**

- Domain: Evaluation
- Description: Establish the starting-point accuracy before any fine-tuning.
- Sub-Tasks:
  - [x] Run `whisper-small-khmer-v2` on the full eval set
  - [x] Compute CER and WER using the Card 6 utility
  - [x] Save 5–10 example transcriptions (both good and bad) to `docs/baseline_results.md`
  - [x] Record final baseline numbers at the top of `docs/baseline_results.md`

**Card 8: `[DATA-04] Log Dataset Sources & Licenses`**

- Domain: Data
- Description: Track every dataset used so licensing/sourcing is documented from day one.
- Sub-Tasks:
  - [x] In `data/DATASETS.md`, add rows for FLEURS, OpenSLR SLR42, km-speech-corpus (mark as "already in baseline")
  - [x] Add candidate rows for DDD-Cambodia dataset and any others under consideration
  - [x] Include columns: Dataset | Source | License | Hours | Status

**Card 9: `[DATA-05] Acquire DDD-Cambodia Dataset`**

- Domain: Data
- Description: Download and verify the primary additional training dataset.
- Sub-Tasks:
  - [x] Download the DDD-Cambodia Khmer ASR Cultural Dataset
  - [x] Verify license terms (CC-BY-SA) and note attribution requirements
  - [x] Store raw files in `data/raw/`
  - [x] Update hour count in `data/DATASETS.md`

**Card 10: `[DATA-06] Collect Real-World Test Sample`**

- Domain: Data
- Description: Gather real, spontaneous audio for evaluation only — never for training.
- Sub-Tasks:
  - [x] Record or collect ~15–20 real Khmer voice notes (phone mic, casual speech)
  - [x] Get consent/permission from anyone recorded
  - [x] Transcribe them accurately as ground truth
  - [x] Add to `data/eval/` (separate subfolder, e.g. `data/eval/real_world/`)

**Card 11: `[DATA-07] Clean & Format DDD-Cambodia Data`**

- Domain: Data
- Description: Prepare the new dataset for training.
- Sub-Tasks:
  - [x] Align audio files with their transcripts
  - [x] Resample all audio to 16kHz mono `.wav`
  - [x] Store cleaned output in `data/processed/`
  - [x] Spot-check a sample of aligned pairs for correctness

**Card 12: `[DATA-08] Build Audio Augmentation Pipeline`**

- Domain: Data
- Description: Generate synthetic "phone-mic quality" training data from clean audio.
- Sub-Tasks:
  - [x] Install `audiomentations`
  - [x] Write augmentation script in `notebooks/02_data_prep.ipynb` (background noise, static, bitrate/gain reduction)
  - [x] Generate augmented copies into `data/processed/`
  - [x] Listen to a few samples to sanity-check they sound like real phone audio, not pure noise

**Card 13: `[DATA-09] Merge & Finalize Training Dataset`**

- Domain: Data
- Description: Combine all data sources into the final training set.
- Sub-Tasks:
  - [x] Merge FLEURS + OpenSLR + km-speech-corpus (inherited) with DDD-Cambodia + augmented data
  - [x] Deduplicate and shuffle
  - [x] Document final composition (source breakdown, total hours) in `data/DATASETS.md`

**Card 14: `[ML-01] Set Up LoRA Fine-Tuning Script`**

- Domain: Training
- Description: Configure the fine-tuning pipeline on top of the existing checkpoint.
- Sub-Tasks:
  - [x] In `notebooks/03_lora_finetune.ipynb`, load `whisper-small-khmer-v2` as the base model
  - [x] Configure PEFT/LoRA adapter (target attention layers)
  - [x] Set training arguments (batch size, learning rate, epochs) suited to T4 GPU memory

**Card 15: `[ML-02] Implement Drive Checkpointing`**

- Domain: Training
- Description: Protect training progress against Colab session drops.
- Sub-Tasks:
  - [x] Save LoRA weights + optimizer state to Google Drive every N steps
  - [x] Write resume-from-checkpoint logic
  - [x] Test by manually interrupting and resuming a short training run

**Card 16: `[ML-03] Run Fine-Tuning`**

- Domain: Training
- Description: Execute the actual training run(s).
- Sub-Tasks:
  - [x] Run training on Colab, monitoring loss and session stability
  - [x] Log training metrics (loss curve, steps completed)

**Card 17: `[EVAL-03] Evaluate Fine-Tuned Model vs. Baseline`**

- Domain: Evaluation
- Description: Measure whether fine-tuning actually improved results.
- Sub-Tasks:
  - [x] Run the fine-tuned model on the same eval set as Card 7
  - [x] Compute CER/WER "Improvement Delta" vs. baseline
  - [x] Log results in `docs/`

**Card 18: `[ML-04] Export Final Model Weights`**

- Domain: Training
- Description: Package the trained adapter for use in the bot.
- Sub-Tasks:
  - [x] Save final LoRA adapter (not full model) to `models/checkpoints/`
  - [x] Write `models/MODEL_CARD.md` (training data, metrics, known limitations)

**Card 19: `[BE-01] Build Audio Ingestion & Conversion`**

- Domain: Backend
- Description: Handle incoming audio files before they reach the model.
- Sub-Tasks:
  - [ ] Implement `src/audio/convert.py` (`.ogg` → 16kHz mono `.wav`)
  - [ ] Implement `src/audio/validate.py` (file type + size checks)
  - [ ] Test with a real Telegram voice note file

**Card 20: `[BE-02] Build ASR Inference Module`**

- Domain: Backend
- Description: Wrap the fine-tuned model for use in the bot backend.
- Sub-Tasks:
  - [ ] Implement `src/asr/model.py` (load base + LoRA adapter)
  - [ ] Implement `src/asr/infer.py` (transcribe a `.wav` file, return text)
  - [ ] Benchmark inference time on a sample clip

**Card 21: `[BE-03] Build Gemini Correction Client`**

- Domain: Backend
- Description: Add the LLM post-processing layer.
- Sub-Tasks:
  - [ ] Implement `src/correction/gemini_client.py`
  - [ ] Write the correction-only system prompt (fix errors, don't change meaning)
  - [ ] Test with a few known-messy ASR outputs

**Card 22: `[BE-04] Build Fallback Correction Path`**

- Domain: Backend
- Description: Ensure the bot doesn't hard-fail if Gemini is unavailable.
- Sub-Tasks:
  - [ ] Implement `src/correction/fallback.py`
  - [ ] Define fallback behavior (return raw ASR text, or use a local model)
  - [ ] Test by simulating a Gemini API failure

**Card 23: `[BOT-01] Wire Up Telegram Bot Handlers`**

- Domain: Bot/Deploy
- Description: Connect all pipeline pieces into a working bot.
- Sub-Tasks:
  - [ ] Implement `src/bot/main.py` (bot entrypoint, token setup)
  - [ ] Implement `src/bot/handlers.py` (voice message → convert → transcribe → correct → reply)
  - [ ] Test end-to-end with a real Telegram account

**Card 24: `[BE-05] Add Error Handling`**

- Domain: Backend
- Description: Handle edge cases gracefully.
- Sub-Tasks:
  - [ ] Handle unsupported file types with a clear user message
  - [ ] Handle oversized files
  - [ ] Handle failed transcription without silent failure

**Card 25: `[BE-06] Write Unit Tests`**

- Domain: Backend
- Description: Cover the core pipeline stages with tests.
- Sub-Tasks:
  - [ ] Write `tests/test_audio.py`
  - [ ] Write `tests/test_asr.py`
  - [ ] Write `tests/test_correction.py`

**Card 26: `[BOT-02] Containerize the App`**

- Domain: Bot/Deploy
- Description: Package the bot for deployment.
- Sub-Tasks:
  - [ ] Write `deployment/Dockerfile`
  - [ ] Build and run the container locally
  - [ ] Confirm the bot responds correctly inside the container

**Card 27: `[BOT-03] Deploy to Hosting Platform`**

- Domain: Bot/Deploy
- Description: Get the bot running in production.
- Sub-Tasks:
  - [ ] Choose Hugging Face Spaces or Render
  - [ ] Configure deployment (`render.yaml` if Render)
  - [ ] Set environment variables (API keys, tokens) securely
  - [ ] Confirm live bot responds end-to-end

**Card 28: `[BOT-04] Test Concurrency`**

- Domain: Bot/Deploy
- Description: Confirm the bot handles multiple users at once.
- Sub-Tasks:
  - [ ] Simulate 2–3 simultaneous voice message submissions
  - [ ] Confirm no blocking or dropped requests
  - [ ] Note Gemini free-tier rate limits as a constraint

**Card 29: `[EVAL-04] Measure End-to-End Latency`**

- Domain: Evaluation
- Description: Confirm the bot meets the speed target.
- Sub-Tasks:
  - [ ] Measure time from audio submission to text delivery
  - [ ] Test with a standard 10-second clip
  - [ ] Confirm under 5-second target; note bottlenecks if not met

**Card 30: `[EVAL-05] Compute LLM Enhancement Delta`**

- Domain: Evaluation
- Description: Quantify the value the Gemini correction layer adds.
- Sub-Tasks:
  - [ ] Compare raw fine-tuned ASR output vs. Gemini-corrected output on the eval set
  - [ ] Compute CER/WER for both
  - [ ] Document the delta clearly (this is a key thesis result)

**Card 31: `[EVAL-06] Full Pipeline Comparison`**

- Domain: Evaluation
- Description: Bring all results together in one place.
- Sub-Tasks:
  - [ ] In `04_eval_comparison.ipynb`, compare baseline vs. fine-tuned vs. fine-tuned+LLM
  - [ ] Run comparison on both FLEURS-style and real-world holdout audio
  - [ ] Export a clean results table

**Card 32: `[DOCS-01] Visualize Results`**

- Domain: Docs/Thesis
- Description: Prepare charts for the write-up and defense.
- Sub-Tasks:
  - [ ] Build CER/WER comparison chart across pipeline stages
  - [ ] Build a chart showing real-world vs. benchmark-style performance
  - [ ] Save exportable images/figures for the thesis document

**Card 33: `[DOCS-02] Finalize Dataset Documentation`**

- Domain: Docs/Thesis
- Description: Turn `DATASETS.md` into a write-up-ready section.
- Sub-Tasks:
  - [ ] Review and clean up `data/DATASETS.md`
  - [ ] Confirm every dataset has source, license, and hour count listed
  - [ ] Draft the "Data" section text for the thesis

**Card 34: `[DOCS-03] Draft Thesis Chapters`**

- Domain: Docs/Thesis
- Description: Write up the project for submission.
- Sub-Tasks:
  - [ ] Draft Methodology chapter using `docs/thesis_notes/`
  - [ ] Draft Results chapter using Card 31/32 outputs
  - [ ] Draft Discussion/Limitations chapter

**Card 35: `[ADV-01] Get Proposal Signed`**

- Domain: Advisor
- Description: Finalize approval before deep implementation work continues.
- Sub-Tasks:
  - [ ] Share proposal doc with advisor and supervisor
  - [ ] Incorporate any requested changes
  - [ ] Get signature and confirm final start/end dates

**Card 36: `[ADV-02] Periodic Advisor Check-Ins`**

- Domain: Advisor
- Description: Recurring task — duplicate or reuse this card at each phase milestone.
- Sub-Tasks:
  - [ ] Send a short progress update message
  - [ ] Share any blockers or decisions needing advisor input
  - [ ] Note advisor feedback for the next cycle
