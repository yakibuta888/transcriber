import json
import torch

from transformers.pipelines import pipeline


model_id = "kotoba-tech/kotoba-whisper-v2.2"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}

pipe = pipeline(
    model=model_id,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs=model_kwargs,
    batch_size=8,
    trust_remote_code=True,
)

result = pipe("sample_diarization_japanese.mp3", chunk_length_s=15, add_punctuation=True)

with open("transcription.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)