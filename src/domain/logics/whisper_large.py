import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines import pipeline

from domain.common.progress_reporter import ProgressReporter


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


    def transcribe(self, audio: dict, language: str = "ja", progress: ProgressReporter | None = None):
        """
        音声認識を実行し、進捗を報告する
        - progress: UnifiedProgressReporterインスタンス（進捗報告用）
        """
        generate_kwargs = {
            "language": language
        }
        
        # 進捗開始通知
        if progress:
            progress.update.transcription(0, "音声認識を開始")
        
        result = self.pipe(audio, generate_kwargs=generate_kwargs)

        if result is None:
            return []
        
        if isinstance(result, dict) and "chunks" in result:
            if progress:
                progress.update.transcription(1, "音声認識が完了")
            return result["chunks"]
        else:
            return []
