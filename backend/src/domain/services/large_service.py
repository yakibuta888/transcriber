import os
import torchaudio

from concurrent.futures import ThreadPoolExecutor

from src.config import HUGGING_FACE_TOKEN
from src.domain.common.get_models_dir import get_models_path
from src.domain.exception.could_not_diarize_error import CouldNotDiarizeError
from src.domain.entity.audio_entity import AudioEntity
from src.domain.interfaces.progress_reporter import IProgressReporter
from src.domain.interfaces.transcriber import ITranscriber
from src.domain.logics.merger import ResultMerger
from src.domain.logics.speaker_diarizer import SpeakerDiarizer
from src.domain.logics.whisper_large import WhisperLargeTranscriber
from src.domain.services.pre_processing_service import PreprocessingService
from src.settings import logger


class LargeService(ITranscriber):
    def __init__(self, audio_file: str, diarizer_model_id: str, whisper_model_id: str, hf_token: str | None = None):
        # 初期化時のバリデーションとデフォルト値設定
        if os.path.exists(audio_file):
            self.audio_file = audio_file
        else:
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # diarizerモデルの準備
        if not diarizer_model_id:
            diarizer_model_id = "speaker-diarization-3.1"
        try:
            self.diarizer_model = get_models_path(os.path.join(diarizer_model_id, "config.yaml"))
            logger.info(f"Using diarizer model: {self.diarizer_model}")
        except FileNotFoundError as e:
            self.diarizer_model = f"pyannote/{diarizer_model_id}"
            logger.warning(f"Local diarizer model not found, using default: {self.diarizer_model}. \nError: {e}")

        # whisper準備
        if not whisper_model_id:
            whisper_model_id = "whisper-large-v3"
        try:
            self.whisper_model = get_models_path(whisper_model_id)
            logger.info(f"Using whisper model: {self.whisper_model}")
        except FileNotFoundError as e:
            self.whisper_model = f"openai/{whisper_model_id}"
            logger.warning(f"Local whisper model not found, using default: {self.whisper_model}. \nError: {e}")

        if hf_token:
            self.hf_token = hf_token
        elif HUGGING_FACE_TOKEN:
            self.hf_token = HUGGING_FACE_TOKEN
        else:
            raise ValueError("Hugging Face token is required for accessing models.")


    def run(self, option_args: dict, progress: IProgressReporter | None = None) -> list[dict]:
        diarize: bool = option_args.get("diarize", True)
        chunk_length_s: int = option_args.get("chunk_length", 15)
        if chunk_length_s < 5:
            chunk_length_s = 5
            print('\033[31m' + "Chunk length is too short, setting to 5 seconds. Please set a value between 5 and 180 seconds." + '\033[0m')
            logger.warning("Chunk length is too short, setting to 5 seconds.")
        elif chunk_length_s > 180:
            chunk_length_s = 180
            print('\033[31m' + "Chunk length is too long, setting to 180 seconds. Please set a value between 5 and 180 seconds." + '\033[0m')
            logger.warning("Chunk length is too long, setting to 180 seconds.")
        else:
            logger.debug(f"Chunk length is set to {chunk_length_s} seconds.")

        if progress:
            progress.set_totals(
                preprocessing=6,
                diarization=1 if diarize else 0,
                transcription=1,
                merge=1 if diarize else 0,  # 後で設定
                output=1  # 後で設定
            )

        try:
            # 音声読み込みと前処理
            pre_processing_service = PreprocessingService()
            audio_entity: AudioEntity = pre_processing_service.process(self.audio_file, progress=progress)

            # 並列実行
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Whisper: セグメント全体を一括で音声認識
                def asr_task():
                    transcriber = WhisperLargeTranscriber(
                        self.whisper_model,
                        diarize=diarize,
                        chunk_length_s=chunk_length_s,
                        batch_size=option_args.get("batch_size", 8),
                        flash_attention=option_args.get("flash_attention", False),
                    )
                    audio_array = audio_entity.for_whisper()["array"]
                    sampling_rate = audio_entity.for_whisper()["sampling_rate"]
                    chunk_samples = int(chunk_length_s * sampling_rate)
                    
                    chunks = [
                        audio_array[i:i + chunk_samples]
                        for i in range(0, len(audio_array), chunk_samples)
                    ]
                    
                    # 進捗開始通知
                    if progress:
                        steps_info = {
                            "total": len(chunks),
                            "current": 0,
                            "detail": f"/{len(chunks)}:音声認識中..."
                        }
                        progress.set_transcribe_total(len(chunks))
                        progress.update.transcription(0, "音声認識を開始")

                    final_transcript: list[dict] = []
                    for idx, chunk in enumerate(chunks):
                        steps_info["current"] = idx + 1
                        offset_sec = idx * chunk_length_s  # チャンクの開始位置（全体音声に対して何秒目か）
                        chunk_audio = {
                            "array": chunk,
                            "sampling_rate": sampling_rate,
                        }
                        chunk_results = transcriber.transcribe(chunk_audio, steps_info=steps_info)  # Whisperから得られるlist[dict]
                        for seg in chunk_results:
                            if isinstance(seg, str):
                                seg = {
                                    "timestamp": [0, chunk_length_s],
                                    "text": seg
                                }
                            # チャンク内のtimestampを絶対時刻に補正
                            abs_start = seg['timestamp'][0] + offset_sec if seg['timestamp'][0] is not None else offset_sec
                            abs_end = seg['timestamp'][1] + offset_sec if seg['timestamp'][1] is not None else chunk_length_s + offset_sec
                            final_transcript.append({
                                'timestamp': [abs_start, abs_end],
                                'text': seg['text'],
                            })

                    if progress:
                        progress.update.transcription(steps_info["current"], "音声認識が完了")

                    return final_transcript

                # Diarization: 話者分離
                def diar_task():
                    diarizer = SpeakerDiarizer(self.diarizer_model, self.hf_token)
                    return diarizer.get_segments(
                        audio_entity.for_pyannote(),
                        option_args.get("num_speakers", None),
                        option_args.get("min_speakers", None),
                        option_args.get("max_speakers", None),
                        progress=progress
                    )

                future_asr = executor.submit(asr_task)
                future_diar = executor.submit(diar_task) if diarize else None
                asr_segments = future_asr.result()
                diar_segments = future_diar.result() if future_diar else None

            if not asr_segments or (diarize and not diar_segments):
                logger.warning("No segments detected. Exiting transcription. @LargeService.run")
                raise CouldNotDiarizeError("No segments detected. Please check the audio file or models.")

            logger.debug(f"ASR segments: {len(asr_segments)}, Diarization segments: {len(diar_segments or [])}")
            
            if diarize:
                # 話者情報付与前に正確な数を設定
                if progress:
                    progress.set_merge_total(len(asr_segments))
                
                # セグメントごとに話者情報を付与
                results = ResultMerger.merge(asr_segments, diar_segments, progress=progress)
                logger.debug("Transcription with speaker attribution completed.")
                return results
            else:
                results = [{
                    'start': seg['timestamp'][0],
                    'end': seg['timestamp'][1],
                    'speaker': "Unknown",
                    'text': seg['text'],
                } for seg in asr_segments]
                return results

        except Exception as e:
            logger.error(f"An error occurred during transcription. @LargeService.run: {e}", exc_info=True)
            raise e