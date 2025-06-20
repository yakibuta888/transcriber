from domain.common.output import write_text_file
from domain.common.progress_reporter import ProgressReporter
from settings import logger

class TranscribeService:
    def __init__(self, audio_file: str, model: str, hf_token: str):
        match model:
            case "whisper-large-v3":
                from domain.services.large_service import LargeService
                self.transcriber = LargeService(
                    audio_file=audio_file,
                    diarizer_model_id="speaker-diarization-3.1",
                    whisper_model_id="whisper-large-v3",
                    hf_token=hf_token
                )
            case _:
                raise ValueError(f"Unsupported model: {model}")

    def transcribe_and_save(self, outdir: str, outname: str, option_args: dict, progress: ProgressReporter | None = None):
        try:
            results = self.transcriber.run(option_args=option_args, progress=progress)
            
            # ファイル出力の進捗表示（set_totalsは呼ばない）
            if progress:
                # 出力行数を後から設定する方法を追加
                progress.set_output_total(1)
                progress.update.output(0, "音声認識結果を出力中...")
                write_text_file(outdir, outname, results)
                progress.update.output(1, "音声認識結果の出力が完了しました。")
            else:
                write_text_file(outdir, outname, results)
                
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")