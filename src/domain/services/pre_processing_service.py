import os
from domain.common.progress_reporter import ProgressReporter
from domain.entity.audio_entity import AudioEntity
from domain.logics.audio_loader import AudioLoader
from domain.logics.audio_normalizer import AudioNormalizer
from domain.logics.audio_processor import AudioProcessor
from domain.logics.transient_noise_reducer import TransientNoiseReducer
from domain.logics.vocal_enhancer import VocalEnhancer


class PreprocessingService:
    def __init__(self):
        self._stationary_processor = AudioProcessor()
        self._transient_processor = TransientNoiseReducer(
            rnn_model_path=os.path.join("src", "models", "CleanUNet", "exp", "DNS-large-full", "checkpoint", "pretrained.pkl")
        )
        self._enhancer = VocalEnhancer(
            noise_reduce=False,  # 非定常ノイズ除去はPreprocessingServiceで行うため無効化
            eq_settings={'low_gain': 1.1, 'high_gain': 1.4}
        )
        self._steps = list()  # 音声処理のステップ
    

    def process(self, path: str, progress: ProgressReporter | None = None) -> AudioEntity:
        # Step 1: ファイル読み込みとリサンプリング、モノラル変換
        if progress:
            progress.update.preprocessing(0, "音声ファイルを読み込み中...")
        raw_audio = AudioLoader.load(path)
        self._steps.append("resample_and_convert_to_mono")

        # Step 2: 定常ノイズ除去
        if progress:
            progress.update.preprocessing(1, "定常ノイズ除去中...")
        clean_audio = self._stationary_processor.reduce_stationary_noise(raw_audio.waveform)
        self._steps.append("reduce_stationary_noise")

        # Step 3: ピーク正規化
        if progress:
            progress.update.preprocessing(2, "ピーク正規化中...")
        normalized_audio = AudioNormalizer.peak_normalize(clean_audio)
        self._steps.append("peak_normalize")

        # Step 4: RNNoiseによる非定常ノイズ除去（追加処理）
        if progress:
            progress.update.preprocessing(3, "非定常ノイズ除去中...")
        denoised_audio = self._transient_processor.reduce(normalized_audio)
        self._steps.append("reduce_transient_noise")
        
        # Step 5: ボーカルエンハンスメント
        if progress:
            progress.update.preprocessing(4, "ボーカルエンハンスメント中...")
        enhanced_audio = self._enhancer.enhance(denoised_audio)
        self._steps.append("vocal_enhancement")
        
        # Step 6: 再正規化
        if progress:
            progress.update.preprocessing(5, "再正規化中...")
        re_normalized_audio = AudioNormalizer.peak_normalize(enhanced_audio)
        self._steps.append("re_normalize")

        # Step 7: AudioEntityの作成
        if progress:
            progress.update.preprocessing(6, "音声ファイルの読み込みが完了しました。")
        return AudioEntity.new(
            waveform=re_normalized_audio,
            sample_rate=raw_audio.sample_rate,
            duration=raw_audio.duration,
            num_channels=re_normalized_audio.size(0) if re_normalized_audio.dim() > 1 else 1,
            processing_steps=self._steps,
            metadata={'source': raw_audio.path}
        )


    @property
    def steps(self):
        """音声処理のステップ数を返す（進捗表示用）"""
        return self._steps