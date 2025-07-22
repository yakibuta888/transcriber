import os

from config import HUGGING_FACE_TOKEN
from domain.common.get_models_dir import get_models_path
from domain.common.progress_reporter import ProgressReporter
from domain.exception.could_not_transcribe_error import CouldNotTranscribeError
from domain.interfaces.transcriber import ITranscriber
from domain.logics.kotoba_whisper_v2 import KotobaWhisperTranscriber
from settings import logger


class KotobaWhisperService(ITranscriber):
    def __init__(self, audio_file: str, model_id: str, hf_token: str | None = None):
        """
        Kotoba Whisper Service for audio transcription.

        Args:
            audio_file (str): Path to the audio file to be transcribed.
            model_id (str): Model identifier for the transcription.
            hf_token (str | None): Hugging Face token for accessing the model.
        """
        # 初期化時のバリデーションとデフォルト値設定
        if os.path.exists(audio_file):
            self.audio_file = audio_file
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # モデルの準備
        if not model_id:
            model_id = "kotoba-whisper-v2.2"
        try:
            self.kotoba_whisper_model = get_models_path(model_id)
            logger.info(f"Using Kotoba Whisper model: {self.kotoba_whisper_model}")
        except FileNotFoundError as e:
            self.kotoba_whisper_model = f"kotoba-tech/{model_id}"
            logger.warning(f"Local Kotoba Whisper model not found, using default: {self.kotoba_whisper_model}. \nError: {e}")


        if hf_token:
            self.hf_token = hf_token
        elif HUGGING_FACE_TOKEN:
            self.hf_token = HUGGING_FACE_TOKEN
        else:
            raise ValueError("Hugging Face token is required for accessing models.")


    def run(self, option_args: dict, progress: ProgressReporter | None = None) -> list[dict]:
        # 進捗開始通知
        if progress:
            progress.set_totals(
                preprocessing=0,
                diarization=0,
                transcription=1,
                merge=0,
                output=1
            )

        try:
            # Kotoba Whisperのインスタンスを作成
            transcriber = KotobaWhisperTranscriber(
                model=self.kotoba_whisper_model,
                hf_token=self.hf_token,
                chunk_length_s=option_args.get("chunk_length", 15),
                batch_size=option_args.get("batch_size", 8),
                flash_attention=option_args.get("flash_attention", False)
            )

            # 音声認識を実行
            result = transcriber.transcribe(
                audio_file=self.audio_file,
                add_punctuation=option_args.get("add_punctuation", True),
                num_speakers=option_args.get("num_speakers", None),
                min_speakers=option_args.get("min_speakers", None),
                max_speakers=option_args.get("max_speakers", None),
                add_silence_start=option_args.get("add_silence_start", None),
                add_silence_end=option_args.get("add_silence_end", None),
                progress=progress
            )
            
            if not result:
                raise CouldNotTranscribeError("No transcription results found.@KotobaWhisperService.run")

            logger.debug(f"Transcription segments: {len(result)}")
            
            return result
        except Exception as e:
            logger.error(f"Transcription failed.@KotobaWhisperService.run: {e}", exc_info=True)
            raise e