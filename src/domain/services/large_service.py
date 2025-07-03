import os
import torchaudio

from concurrent.futures import ThreadPoolExecutor

from config import HUGGING_FACE_TOKEN
from domain.common.progress_reporter import ProgressReporter
from domain.common.get_models_dir import get_models_path
from domain.exception.could_not_diarize_error import CouldNotDiarizeError
from domain.entity.audio_entity import AudioEntity
from domain.logics.merger import ResultMerger
from domain.logics.speaker_diarizer import SpeakerDiarizer
from domain.logics.whisper_large import WhisperTranscriber
from domain.services.pre_processing_service import PreprocessingService
from settings import logger


class LargeService:
    def __init__(self, audio_file: str, diarizer_model_id: str, whisper_model_id: str, hf_token: str):
        # 初期化時のバリデーションとデフォルト値設定
        if os.path.exists(audio_file):
            self.audio_file = audio_file
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        if diarizer_model_id:
            self.diarizer_model_id = diarizer_model_id
        else:
            self.diarizer_model_id = "speaker-diarization-3.1"

        if whisper_model_id:
            self.whisper_model_id = whisper_model_id
        else:
            self.whisper_model_id = "whisper-large-v3"

        if hf_token:
            self.hf_token = hf_token
        elif HUGGING_FACE_TOKEN:
            self.hf_token = HUGGING_FACE_TOKEN
        else:
            raise ValueError("Hugging Face token is required for accessing models.")


    def run(self, option_args: dict, progress: ProgressReporter | None = None):
        try:
            # 音声読み込みと前処理
            pre_processing_service = PreprocessingService()
            audio_entity = pre_processing_service.process(self.audio_file, progress=progress)
            raise UserWarning("Preprocessing is complete. (debugging)")

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

            if progress:
                progress.set_totals(
                    preprocessing_steps=len(pre_processing_service.steps),
                    diar_segments=1,
                    asr_chunks=1,
                    merge_segments=0,  # 後で設定
                    output_lines=0  # 後で設定
                )

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
                    return transcriber.transcribe(audio_entity.for_whisper(), progress=progress)

                # Diarization: 話者分離
                def diar_task():
                    diarizer = SpeakerDiarizer(diarizer_model, self.hf_token)
                    return diarizer.get_segments(
                        audio_entity.for_pyannote(),
                        option_args.get("num_speakers", None),
                        option_args.get("min_speakers", None),
                        option_args.get("max_speakers", None),
                        progress=progress
                    )

                future_asr = executor.submit(asr_task)
                future_diar = executor.submit(diar_task)
                asr_segments = future_asr.result()
                diar_segments = future_diar.result()

            if not asr_segments or not diar_segments:
                logger.warning("No segments detected. Exiting transcription. @LargeService.run")
                raise CouldNotDiarizeError("No segments detected. Please check the audio file or models.")

            logger.debug(f"ASR segments: {len(asr_segments)}, Diarization segments: {len(diar_segments)}")
            
            # 話者情報付与前に正確な数を設定
            if progress:
                progress.set_merge_total(len(asr_segments))
            
            # セグメントごとに話者情報を付与
            results = ResultMerger.merge(asr_segments, diar_segments, progress=progress)
            logger.debug("Transcription with speaker attribution completed.")
            return results

        except Exception as e:
            logger.error(f"An error occurred during transcription. @LargeService.run: {e}", exc_info=True)
            raise e