"""Load the fine-tuned Khmer Whisper (base checkpoint + LoRA adapter) for inference.

Load ONCE at bot startup (get_asr()) - loading takes many seconds and the model is big.
torch / transformers / peft are imported lazily so importing this module stays cheap.
"""
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..audio.validate import AudioError

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "seanghay/whisper-small-khmer-v2"
DEFAULT_ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "models/checkpoints/final_adapter")
LANGUAGE = "km"
TASK = "transcribe"


class ASRError(AudioError):
    """Model loading / transcription problem. `.user_message` is safe to show the user."""


@dataclass
class ASRModel:
    model: Any
    processor: Any
    device: str
    prompt_length: int          # number of forced decoder prompt tokens (lang/task/notimestamps)
    fp16: bool = False
    # generate() on one shared model is not safe to run from several threads at once (FR-8)
    lock: threading.Lock = field(default_factory=threading.Lock)


def check_adapter_trained(peft_model, min_norm: float = 1e-6) -> float:
    """Guard against the untrained-adapter bug (see docs/results.md postmortem).

    LoRA initialises every lora_B matrix to exactly zero, so an adapter whose lora_B
    weights are all still zero does nothing - the model silently behaves like the base.
    Returns the total lora_B norm; raises ASRError if the adapter is missing or untrained.
    """
    total, count = 0.0, 0
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            total += param.detach().float().norm().item()
            count += 1

    if count == 0:
        raise ASRError(
            "The speech model failed to load correctly. Please try again later.",
            detail="no lora_B parameters found - adapter layers were not injected",
        )
    if total < min_norm:
        raise ASRError(
            "The speech model failed to load correctly. Please try again later.",
            detail=f"all {count} lora_B tensors are zero (norm={total}) - adapter is UNTRAINED; "
                   "re-copy the real checkpoint into the adapter dir",
        )
    return total


def load_asr_model(
    adapter_dir: Optional[str] = None,
    base_model_id: str = BASE_MODEL_ID,
    device: Optional[str] = None,
    fp16: bool = False,
    merge: bool = True,
) -> ASRModel:
    """Load base model + LoRA adapter.

    fp16:  half precision on GPU - faster, but numerics differ slightly from the fp32 run
           that produced the reported CER. Default off so deployment == what was evaluated.
    merge: fold the LoRA weights into the base weights (slightly faster, no adapter overhead).
    """
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_path = Path(adapter_dir or DEFAULT_ADAPTER_DIR)
    has_weights = (adapter_path / "adapter_model.safetensors").is_file() or \
                  (adapter_path / "adapter_model.bin").is_file()
    if not (adapter_path / "adapter_config.json").is_file() or not has_weights:
        raise ASRError(
            "The speech model is not available right now. Please try again later.",
            detail=f"adapter files not found in {adapter_path.resolve()}",
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s + adapter %s on %s", base_model_id, adapter_path, device)

    base = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))

    norm = check_adapter_trained(peft_model)          # fail loudly BEFORE merging/serving
    logger.info("Adapter OK (total lora_B norm = %.3f)", norm)

    model = peft_model.merge_and_unload() if merge else peft_model
    if fp16 and device.startswith("cuda"):
        model = model.half()
    else:
        fp16 = False
    model = model.to(device).eval()

    # The processor is not changed by LoRA training, so load it from the base checkpoint.
    processor = WhisperProcessor.from_pretrained(base_model_id)

    # Same decoding setup that produced the evaluated CER (03/04 notebooks).
    prompt_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.generation_config.forced_decoder_ids = prompt_ids

    return ASRModel(model=model, processor=processor, device=device,
                    prompt_length=len(prompt_ids), fp16=fp16)


_ASR: Optional[ASRModel] = None
_ASR_LOCK = threading.Lock()


def get_asr(**load_kwargs) -> ASRModel:
    """Process-wide singleton: the first call loads the model, later calls reuse it."""
    global _ASR
    with _ASR_LOCK:
        if _ASR is None:
            _ASR = load_asr_model(**load_kwargs)
        return _ASR