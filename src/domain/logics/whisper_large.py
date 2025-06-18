import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines import pipeline


class WhisperTranscriber:
    def __init__(self, model: str, chunk_length_s: int = 15, batch_size: int = 8, flash_attention: bool = False):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # attn_implementationの設定
        if flash_attention:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"
        
        processor = AutoProcessor.from_pretrained(model)
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation=attn_impl if torch.cuda.is_available() else None
        )

        # pipeline の設定
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=asr_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            return_timestamps=True,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            trust_remote_code=True,
        )


    def transcribe(self, wav_path: str, language: str = "ja"):
        generate_kwargs = {
            "language": language
        }
        result = self.pipe(wav_path, generate_kwargs=generate_kwargs)

        if result is None:
            return ""
        
        # Handle the case where result might be a dict or have different structure
        if isinstance(result, dict) and "text" in result:
            return str(result.get("text", "")).strip()
        else:
            return str(result).strip()
