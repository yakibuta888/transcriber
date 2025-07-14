import torch
from transformers.pipelines import pipeline

from domain.common.progress_reporter import ProgressReporter



class KotobaWhisperTranscriber:
    def __init__(self, model: str, hf_token: str, chunk_length_s: int = 15, batch_size: int = 8, flash_attention: bool = False):
        """
        Kotoba Whisper Transcriber for audio transcription.
        
        Args:
            model (str): Model name or path.
            chunk_length_s (int): Length of audio chunks in seconds. Defaults to 15.
            batch_size (int): Batch size for processing. Defaults to 8.
            flash_attention (bool): Whether to use flash attention. Defaults to False.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # attn_implementationの設定
        if flash_attention and self.device == "cuda":
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"

        model_kwargs = {"attn_implementation": attn_impl} if self.device == "cuda" else {}

        self.pipe = pipeline(
            model=model,
            token=hf_token,
            device=self.device,
            torch_dtype=torch_dtype,
            model_kwargs=model_kwargs,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            trust_remote_code=True,
        )
        

    def transcribe(self, audio_file: str, add_punctuation: bool = True, num_speakers: int | None = None, min_speakers: int | None = None, max_speakers: int | None = None, add_silence_start: float | None = None, add_silence_end: float | None = None, progress: ProgressReporter | None = None):

        # pipeの引数として渡す辞書を作成
        pipe_kwargs = {
            "add_punctuation": add_punctuation,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "add_silence_start": add_silence_start,
            "add_silence_end": add_silence_end,
        }

        # None,空文字の値は渡さない
        pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v not in [None, ""]}
        
        # 進捗開始通知
        if progress:
            progress.update.transcription(0, "音声認識を開始")

        result = self.pipe(audio_file, **pipe_kwargs)

        result_list = []
        if isinstance(result, dict) and "chunks" in result:
            for chunk in result['chunks']:
                result_list.append({
                    "speaker": str(chunk.get("speaker_id", "unknown")),
                    "text": str(chunk.get("text", "")),
                    "start": float(chunk["timestamp"][0]) if "timestamp" in chunk else 0.0,
                    "end": float(chunk["timestamp"][1]) if "timestamp" in chunk else 0.0
                })
            
            if progress:
                progress.update.transcription(1, "音声認識が完了")
            
            return result_list
        else:
            raise ValueError("Unexpected result format from transcription pipeline.@KotobaWhisperTranscriber.transcribe")
