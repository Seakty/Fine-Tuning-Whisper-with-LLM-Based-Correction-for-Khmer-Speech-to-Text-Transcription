## Part 2: Project Structure

```
khmer-stt-bot/
├── README.md
├── requirements.txt
├── .env.example                  # Gemini API key, Telegram bot token (never commit real .env)
├── .gitignore
│
├── data/
│   ├── raw/                      # Original downloaded datasets (FLEURS, OpenSLR, etc.)
│   ├── processed/                # Cleaned/resampled audio + aligned transcripts
│   ├── eval/                     # Held-out evaluation set (kept separate, never trained on)
│   └── DATASETS.md               # Source, license, hour count for every dataset used
│
├── notebooks/
│   ├── 01_baseline_eval.ipynb    # Benchmark whisper-small-khmer-v2 as-is
│   ├── 02_data_prep.ipynb        # Merge/clean additional data
│   ├── 03_lora_finetune.ipynb    # Colab training notebook (with Drive checkpointing)
│   └── 04_eval_comparison.ipynb  # CER/WER comparison: baseline vs. fine-tuned vs. +LLM
│
├── src/
│   ├── audio/
│   │   ├── convert.py            # .ogg -> 16kHz .wav conversion
│   │   └── validate.py           # File type/size checks (FR-7)
│   ├── asr/
│   │   ├── model.py              # Load/wrap the fine-tuned Whisper + LoRA model
│   │   └── infer.py              # Run transcription on a .wav file
│   ├── correction/
│   │   ├── gemini_client.py      # Gemini API call + prompt template
│   │   └── fallback.py           # Fallback correction path (FR-5)
│   ├── bot/
│   │   ├── handlers.py           # Telegram message handlers
│   │   └── main.py               # Bot entrypoint
│   └── utils/
│       ├── metrics.py            # CER/WER calculation
│       └── logging_config.py
│
├── models/
│   ├── checkpoints/               # LoRA adapter weights (not full model — keep repo light)
│   └── MODEL_CARD.md              # What this model is trained on, known limitations
│
├── tests/
│   ├── test_audio.py
│   ├── test_asr.py
│   └── test_correction.py
│
├── deployment/
│   ├── Dockerfile
│   └── render.yaml                 # or huggingface Spaces config
│
└── docs/
    ├── requirements_analysis.md    # this document
    ├── project_architecture.md     # the architecture doc from before
    └── thesis_notes/                # working notes for the write-up
```

### Notes on the structure

- **`notebooks/` vs `src/`:** Notebooks are for experimentation/training on Colab; `src/` is the clean, importable code the actual bot runs on. Don't let bot logic live only in a notebook — it won't deploy.
- **`models/checkpoints/` holds LoRA adapters only**, not the full fine-tuned model — much smaller, faster to push to Drive/HF Hub, and keeps the repo from bloating.
- **`data/eval/` is intentionally separate** from `data/raw/` and `data/processed/` — this is what keeps your CER/WER numbers honest and defensible when you present them.
- **`docs/DATASETS.md`** matters more than it looks — for the thesis write-up you'll need a clean record of exactly what data went in and under what license, and it's much easier to keep that updated as you go than to reconstruct later.
