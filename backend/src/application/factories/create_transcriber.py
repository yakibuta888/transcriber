from domain.interfaces.transcriber import ITranscriber


def create_transcriber(audio_file: str, model: str, hf_token: str | None = None) -> ITranscriber:
    """
    Create a transcriber instance based on the provided parameters.
    
    :param audio_file: Path to the audio file to be transcribed.
    :param model: The model to be used for transcription.
    :param hf_token: Hugging Face token for accessing the model.
    :return: An instance of the transcriber.
    """
    match model:
        case "whisper-large-v3":
            from domain.services.large_service import LargeService
            return LargeService(
                audio_file=audio_file,
                diarizer_model_id="speaker-diarization-3.1",
                whisper_model_id="whisper-large-v3",
                hf_token=hf_token
            )
        case "kotoba-whisper-v2.2":
            from domain.services.kotoba_whisper_service import KotobaWhisperService
            return KotobaWhisperService(
                audio_file=audio_file,
                model_id="kotoba-whisper-v2.2",
                hf_token=hf_token
            )
        case _:
            raise ValueError(f"Unsupported model: {model}")