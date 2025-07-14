from domain.common.output import write_text_file
from domain.common.progress_reporter import ProgressReporter
from domain.interfaces.transcriber import ITranscriber
from settings import logger

class TranscribeService:
    def __init__(self, transcriber: ITranscriber):
        self.transcriber = transcriber

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
            raise RuntimeError(f"Transcription failed.@TranscribeService.transcribe_and_save: {e}")