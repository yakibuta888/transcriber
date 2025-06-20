import torch

from pyannote.audio import Pipeline as DiarizationPipeline

from domain.common.progress_reporter import ProgressReporter


class SpeakerDiarizer:
    def __init__(self, model: str, hf_token: str):
        self.pipeline = DiarizationPipeline.from_pretrained(
            model,
            use_auth_token=hf_token
        )
        self.pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def get_segments(self, audio: dict, num_speakers: int | None = None, min_speakers: int | None = None, max_speakers: int | None = None, progress: ProgressReporter | None = None):
        """
        話者分離を実行し、区間情報を返す。
        - num_speakers: 話者数が分かっている場合に指定
        - min_speakers, max_speakers: 話者数の下限・上限を指定
        - progress: UnifiedProgressReporterインスタンス（進捗報告用）
        """
        diarization_args = {}
        if num_speakers is not None:
            diarization_args["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarization_args["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_args["max_speakers"] = max_speakers

        # 進捗開始通知
        if progress:
            progress.update.diarization(0, "話者分離を開始")

        diarization = self.pipeline(audio, **diarization_args)

        segments = []
        segment_count = 0

        # 区間数をカウントし、各区間の情報を収集
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segment_count += 1
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
            
        # 進捗更新
        if progress:
            progress.update.diarization(1, f"話者分離が完了({segment_count} 区間検出)")
        return segments