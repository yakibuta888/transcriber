import os
import torchaudio

from concurrent.futures import ThreadPoolExecutor

from domain.exception.could_not_diarize_error import CouldNotDiarizeError
from domain.common.get_models_dir import get_models_path
from domain.logics.audio_loader import AudioLoader
from domain.logics.merger import ResultMerger
from domain.logics.speaker_diarizer import SpeakerDiarizer
from domain.logics.whisper_large import WhisperTranscriber
from settings import logger

class LargeService:
    def __init__(self, audio_file: str, diarizer_model_id: str, whisper_model_id: str, hf_token: str):
        self.audio_file = audio_file
        self.diarizer_model_id = diarizer_model_id
        self.whisper_model_id = whisper_model_id
        self.hf_token = hf_token

        # 初期化時のバリデーションとデフォルト値設定
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        if not diarizer_model_id:
            self.diarizer_model_id = "speaker-diarization-3.1"
        if not whisper_model_id:
            self.whisper_model_id = "whisper-large-v3"
        if not hf_token:
            self.hf_token = "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"

    def run(self, option_args: dict):
        try:
            # 音声読み込み
            audio = AudioLoader(self.audio_file)

            # diarizerモデルの準備
            try:
                diarizer_model = get_models_path(os.path.join(self.diarizer_model_id, "config.yaml"))
                logger.info(f"Using diarizer model: {diarizer_model}")
            except FileNotFoundError as e:
                diarizer_model = f"pyannote/{self.diarizer_model_id}"
                logger.warning(f"Local diarizer model not found, using default: {diarizer_model}. \nError: {e}")

            # whisper準備
            try:
                whisper_model = get_models_path(self.whisper_model_id)
                logger.info(f"Using whisper model: {whisper_model}")
            except FileNotFoundError as e:
                whisper_model = f"openai/{self.whisper_model_id}"
                logger.warning(f"Local whisper model not found, using default: {whisper_model}. \nError: {e}")

            # 並列実行
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Whisper: セグメント全体を一括で音声認識
                def asr_task():
                    transcriber = WhisperTranscriber(
                        whisper_model,
                        chunk_length_s=option_args.get("chunk_length", 15),
                        batch_size=option_args.get("batch_size", 8),
                        flash_attention=option_args.get("flash_attention", False),
                    )
                    # WhisperのAPIがファイルパス入力の場合
                    return transcriber.transcribe(audio.for_whisper())

                # Diarization: 話者分離
                def diar_task():
                    diarizer = SpeakerDiarizer(diarizer_model, self.hf_token)
                    return diarizer.get_segments(
                        audio.for_pyannote(),
                        option_args.get("num_speakers", None),
                        option_args.get("min_speakers", None),
                        option_args.get("max_speakers", None)
                    )

                future_asr = executor.submit(asr_task)
                future_diar = executor.submit(diar_task)
                asr_segments = future_asr.result()
                diar_segments = future_diar.result()

            if not asr_segments or not diar_segments:
                logger.warning("No segments detected. Exiting transcription. @LargeService.run")
                raise CouldNotDiarizeError("No segments detected. Please check the audio file or models.")

            logger.debug(f"ASR segments: {len(asr_segments)}, Diarization segments: {len(diar_segments)}")
            
            # セグメントごとに話者情報を付与
            results = ResultMerger.merge(asr_segments, diar_segments)
            logger.debug("Transcription with speaker attribution completed.")
            return results

        except Exception as e:
            logger.error(f"An error occurred during transcription. @LargeService.run: {e}")
            raise e