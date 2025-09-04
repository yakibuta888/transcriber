import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines import pipeline

from src.domain.exception.could_not_transcribe_error import CouldNotTranscribeError
from src.domain.interfaces.progress_reporter import IProgressReporter


class WhisperLargeTranscriber:
    def __init__(self, model: str, diarize: bool = True, chunk_length_s: int = 15, batch_size: int = 8, flash_attention: bool = False):
        self.diarize = diarize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # attn_implementationの設定
        if flash_attention and self.device == "cuda":
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"
        
        self.processor = AutoProcessor.from_pretrained(model)
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation=attn_impl if self.device == "cuda" else None
        )

        if diarize:
            # pipeline の設定
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.asr_model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=torch_dtype,
                return_timestamps=True,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                trust_remote_code=True,
            )


    def transcribe(self, audio: dict, language: str = "ja", steps_info: dict | None = None) -> list[dict] | list[str]:
        """
        音声認識を実行し、進捗を報告する
        - steps_info: 進捗情報を含む辞書。例: {"total": 10, "current": 0, "detail": "/10:音声認識中..."}
        """
        if steps_info:
            detail: str = str(steps_info["current"]) + steps_info["detail"]
            print('\r' + detail, end='', flush=True)
            if steps_info["current"] == steps_info["total"]:
                print()

        if self.diarize:
            generate_kwargs = {
                "language": language,
                # "suppress_tokens": [-1],              # 特殊トークン抑制
                # "temperature": (0.0, 0.2, 0.4),     # サンプリング多様性（任意）
            }
            result = self.pipe(audio, generate_kwargs=generate_kwargs)
        else:
            inputs = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            )
            input_features = inputs["input_features"].to(self.device)
            # モデルの型に合わせてキャスト（重要！）
            input_features = input_features.to(self.asr_model.dtype)

            with torch.no_grad():
                # 音声認識を実行
                generated_ids = self.asr_model.generate(
                    input_features,
                    language=language,
                    early_stopping=False,
                )
            
            # 文字起こし結果
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        if result is None:
            return []
        
        if self.diarize and isinstance(result, dict) and "chunks" in result:
            return result["chunks"]
        elif isinstance(result, list):
            return result
        else:
            raise CouldNotTranscribeError("音声認識に失敗しました。結果が不正です。@WhisperLargeTranscriber.transcribe")
