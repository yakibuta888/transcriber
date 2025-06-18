import torch

from pyannote.audio import Pipeline as DiarizationPipeline


class SpeakerDiarizer:
    def __init__(self, model: str, hf_token: str):
        self.pipeline = DiarizationPipeline.from_pretrained(
            model,
            use_auth_token = hf_token
        )
        self.pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def get_segments(self, audio_file: str, num_speakers: int | None = None, min_speakers: int | None = None, max_speakers: int | None = None):
        """
        話者分離を実行し、区間情報を返す。
        - num_speakers: 話者数が分かっている場合に指定
        - min_speakers, max_speakers: 話者数の下限・上限を指定
        """
        diarization_args = {}
        if num_speakers is not None:
            diarization_args["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarization_args["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_args["max_speakers"] = max_speakers

        diarization = self.pipeline(audio_file, **diarization_args)

        return [
            {
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            }
            for segment, _, speaker in diarization.itertracks(yield_label=True)
        ]