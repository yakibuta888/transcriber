from domain.common.output import write_text_file
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

    def transcribe_and_save(self, outdir: str, outname: str, option_args: dict):
        try:
            results = self.transcriber.run(option_args=option_args)
            write_text_file(outdir, outname, results)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")