import numpy as np
import librosa
import torch
import torchaudio
from collections import deque


class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=80, fmax=8000):
        """
        音声処理プロセッサの初期化
        :param sample_rate: サンプリングレート (デフォルト16000Hz)
        :param n_fft: FFTサイズ (デフォルト1024)
        :param hop_length: フレームシフト (デフォルト512)
        :param n_mels: メルバンド数 (デフォルト80)
        :param fmax: 最大周波数 (デフォルト8000Hz)
        """
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels  # 128から80に変更
        self.fmax = fmax      # 8000Hzを明示的に設定
        self.noise_profile = None
        self.adaptive_params = {'beta': 1.5, 'noise_threshold': 0.3}


    def _spectral_analysis(self, audio: np.ndarray) -> dict:
        """
        音声のスペクトル分析を実行
        :param audio: 入力音声信号
        :return: スペクトル特徴量を含む辞書
        """
        # STFT計算
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # メルスペクトログラム計算（fmaxを設定）
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,  # 80に変更
            fmax=self.fmax        # 8000Hzを設定
        )
        
        # ノイズ判定用特徴量抽出
        spectral_flatness = librosa.feature.spectral_flatness(S=mel_spec)  # メルスペクトルを使用
        spectral_centroid = librosa.feature.spectral_centroid(S=mel_spec, sr=self.sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=mel_spec, sr=self.sr)

        return {
            "magnitude": magnitude,
            "phase": phase,
            "mel_spec": mel_spec,
            "flatness": spectral_flatness,
            "centroid": spectral_centroid,
            "bandwidth": spectral_bandwidth
        }


    def _adaptive_spectral_subtraction(self, magnitude: np.ndarray, analysis_data: dict) -> np.ndarray:
        """
        スペクトル分析結果に基づく適応的ノイズ除去
        :param magnitude: 振幅スペクトル
        :param analysis_data: spectral_analysisの出力
        :return: クリーンな振幅スペクトル
        """
        # ノイズタイプ判定
        flatness_mean = np.mean(analysis_data['flatness'])
        centroid_mean = np.mean(analysis_data['centroid'])
        
        # パラメータ動的調整
        beta = self.adaptive_params['beta']
        if flatness_mean > 0.7:  # 定常ノイズ
            beta = 1.8 if centroid_mean < 500 else 1.5
        elif flatness_mean < 0.3:  # 非定常ノイズ
            beta = 1.2
        
        # ノイズプロファイルが未設定の場合、デフォルトを使用
        noise_profile = self.noise_profile if self.noise_profile is not None else np.mean(magnitude, axis=1, keepdims=True)
        
        # スペクトル減算
        magnitude_clean = np.maximum(magnitude - beta * noise_profile, 0)
        return magnitude_clean


    def reduce_stationary_noise(self, audio: torch.Tensor | np.ndarray, noise_ref=None) -> torch.Tensor:
        """
        適応的定常ノイズ除去
        :param audio: 入力音声信号
        :param noise_ref: 外部ノイズ参照（オプション）
        :return: ノイズ除去済み音声
        """
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        # スペクトル分析
        analysis_data = self._spectral_analysis(audio_np)
        
        # ノイズプロファイルの設定
        if noise_ref is not None:
            self._set_noise_profile(noise_ref)
        elif self.noise_profile is None:
            if analysis_data['flatness'].mean() < 0.5:
                noise_samples = int(self.sr * 3.0)  # 3秒のノイズサンプル
            else:
                noise_samples = int(self.sr * 1.0)
            self._set_noise_profile(audio_np[:noise_samples])
        
        # ノイズ閾値チェック
        if np.mean(analysis_data['flatness']) < self.adaptive_params['noise_threshold']:
            return torch.tensor(audio_np, dtype=torch.float32)  # ノイズが少ない場合は処理をスキップ
        
        # 適応的スペクトル減算
        magnitude_clean = self._adaptive_spectral_subtraction(analysis_data['magnitude'], analysis_data)
        
        # 時間領域に復元
        clean_audio = librosa.istft(magnitude_clean * np.exp(1j * analysis_data['phase']), hop_length=self.hop_length)
        return torch.tensor(clean_audio, dtype=torch.float32)

    def _set_noise_profile(self, noise_audio: np.ndarray):
        """
        ノイズプロファイルを設定
        :param noise_audio: ノイズサンプル音声
        """
        noise_stft = librosa.stft(noise_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        noise_mag = np.abs(noise_stft)
        self.noise_profile = np.mean(noise_mag, axis=1, keepdims=True)

    def stream_processing(self, audio_stream, chunk_size=2048):
        """
        ストリーミング処理対応
        :param audio_stream: 音声ストリーム
        :param chunk_size: 処理チャンクサイズ
        :return: 処理済み音声ジェネレータ
        """
        buffer = deque(maxlen=chunk_size)
        for chunk in audio_stream:
            buffer.extend(chunk)
            if len(buffer) >= chunk_size:
                frame = np.array(buffer)
                processed = self.reduce_stationary_noise(frame)
                yield processed
                buffer.clear()

    def update_parameters(self, beta=None, noise_threshold=None):
        """
        パラメータ動的更新
        :param beta: スペクトル減算係数
        :param noise_threshold: ノイズ閾値
        """
        if beta is not None:
            self.adaptive_params['beta'] = beta
        if noise_threshold is not None:
            self.adaptive_params['noise_threshold'] = noise_threshold
