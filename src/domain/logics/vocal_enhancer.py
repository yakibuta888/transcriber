import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from pedalboard._pedalboard import Pedalboard
from pedalboard import Compressor, Gain

from settings import logger


class VocalEnhancer:
    def __init__(
            self,
            sample_rate: int=16000,
            pitch_shift: int=0,
            noise_reduce: bool=True,
            eq_settings: dict={'low_gain': 1.2, 'high_gain': 1.5}
        ):
        """
        ボーカルエンハンスメントクラス
        
        :param sample_rate: サンプリングレート
        :param pitch_shift: ピッチシフト量（半音単位）
        :param noise_reduce: ノイズ除去を有効化
        :param eq_settings: イコライザー設定 {'low_gain': 1.5, 'high_gain': 1.2}
        """
        self.sample_rate = sample_rate
        self.pitch_shift = pitch_shift
        self.noise_reduce = noise_reduce
        self.eq_settings = eq_settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU対応コンポーネント
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={'n_fft': 512, 'hop_length': 256, 'n_mels': 128}
        ).to(self.device)
        
        # MVDRビームフォーミング初期化
        self.mvdr = torchaudio.transforms.MVDR()
        
        # オーディオプロセッシングチェーン
        self.board = Pedalboard([
            Compressor(threshold_db=-20, ratio=3),   # ダイナミクス制御
            Gain(gain_db=2)                          # ゲインブースト
        ])


    def enhance(self, waveform: torch.Tensor) -> torch.Tensor:
        """GPUを活用したボーカルエンハンスメント処理"""
        try:
            # データをGPUに転送
            waveform = waveform.to(self.device)
            
            # 1. ノイズ除去（MVDRビームフォーミング）
            if self.noise_reduce:
                try:
                    waveform = self._apply_mvdr_beamforming(waveform)
                except Exception as e:
                    logger.error(f"MVDRビームフォーミングの適用に失敗しました@VocalEnhancer.enhance: {e}")

            # 2. GPU上でのピッチシフト
            if self.pitch_shift != 0:
                try:
                    waveform = F.pitch_shift(
                        waveform,
                        self.sample_rate,
                        self.pitch_shift, 
                        bins_per_octave=12
                    )
                except Exception as e:
                    logger.error(f"ピッチシフトの適用に失敗しました@VocalEnhancer.enhance: {e}")
            
            # 3. ハーモニックブースト（GPU対応版）
            try:
                waveform = self._gpu_harmonic_boost(waveform)
            except Exception as e:
                logger.error(f"ハーモニックブーストの適用に失敗しました@VocalEnhancer.enhance: {e}")

            # 4. プロフェッショナルEQ処理
            try:
                waveform = self._apply_professional_eq(waveform)
            except Exception as e:
                logger.error(f"EQ処理の適用に失敗しました@VocalEnhancer.enhance: {e}")

            # 5. PyTorchベースのダイナミクス制御
            try:
                waveform = self._pytorch_compressor(waveform, threshold_db=-20, ratio=3)
                waveform = self._pytorch_gain(waveform, gain_db=2)
            except Exception as e:
                logger.error(f"ダイナミクス制御に失敗しました@VocalEnhancer.enhance: {e}")

            # 6. ゲインクリッピング防止（最終出力直前）
            output = torch.clamp(waveform, -1.0, 1.0)

            return output
        except Exception as e:
            logger.error(f"声の強化処理に失敗しました@VocalEnhancer.enhance: {e}")
            return waveform


    def _apply_mvdr_beamforming(self, waveform: torch.Tensor) -> torch.Tensor:
        """MVDRビームフォーミングによるノイズ除去"""
        # マルチチャンネル想定（モノラルを擬似マルチチャンネル化）
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        
        # STFT計算
        stft = torch.stft(waveform, n_fft=512, hop_length=256, return_complex=True)
        
        # ノイズPSD推定（最初の100フレームをノイズサンプルとして使用）
        noise_frames = stft[..., :100]
        psd_noise = torch.einsum('...cft,...dft->...cdft', noise_frames, noise_frames.conj()).mean(-1)
        
        # MVDR適用
        enhanced_stft = self.mvdr(stft, psd_noise)
        return torch.istft(enhanced_stft, n_fft=512, hop_length=256)


    def _safe_detect_pitch(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """
        ピッチ検出の安定性を高めるためのラッパー
        - ピッチ検出失敗時は中央値で補完
        - 検出値が異常な場合はクリッピング
        """
        try:
            # ピッチ検出（torchaudioのdetect_pitch_frequency）
            pitches = F.detect_pitch_frequency(waveform, sample_rate)
            # 無効値除去
            valid_pitches = pitches[(pitches > 50.0) & (pitches < 500.0)]
            if valid_pitches.numel() == 0:
                pitch = 220.0  # デフォルト中央値（A3）
            else:
                pitch = valid_pitches.mean().item()
        except Exception as e:
            pitch = 220.0  # 検出失敗時はデフォルト値
        return pitch


    def _gpu_harmonic_boost(self, waveform: torch.Tensor) -> torch.Tensor:
        """GPU対応版倍音強調処理（ピッチ検出の安定化を反映）"""
        # STFT計算
        stft = torch.stft(waveform, n_fft=512, hop_length=256, return_complex=True)
        mag = torch.abs(stft)

        # 安定化したピッチ検出
        pitch_mean = self._safe_detect_pitch(waveform, self.sample_rate)
        
        if pitch_mean > 0:
            # 倍音周波数のブースト
            harmonics = [pitch_mean * i for i in range(1, 6)]
            freqs = torch.fft.rfftfreq(512, 1/self.sample_rate).to(self.device)
            
            for harmonic in harmonics:
                center_idx = int(torch.argmin(torch.abs(freqs - harmonic)).item())
                start, end = max(0, center_idx-3), min(center_idx+4, mag.shape[-1])
                mag[..., start:end] *= 1.5
        
        # 位相情報と結合
        stft_enhanced = mag * torch.exp(1j * torch.angle(stft))
        return torch.istft(stft_enhanced, n_fft=512, hop_length=256)


    def _apply_professional_eq(self, audio: torch.Tensor) -> torch.Tensor:
        """GPU対応のプロフェッショナルなEQ調整"""
        # audio_low, audio_high を初期化
        audio_low = torch.zeros_like(audio)
        audio_high = torch.zeros_like(audio)

        # ローパスフィルタ（低域ブースト）
        if self.eq_settings['low_gain'] != 1.0:
            audio_low = F.lowpass_biquad(
                audio, self.sample_rate, cutoff_freq=150, Q=0.707
            ) * self.eq_settings['low_gain']
        
        # ハイパスフィルタ（高域ブースト）
        if self.eq_settings['high_gain'] != 1.0:
            audio_high = F.highpass_biquad(
                audio, self.sample_rate, cutoff_freq=5000, Q=0.707
            ) * self.eq_settings['high_gain']
        
        # オリジナルとブレンド（中域を0.7倍に減衰）
        return audio_low + audio_high + (audio * 0.7)


    def _pytorch_compressor(self, audio: torch.Tensor, threshold_db: float = -20, ratio: float = 3.0) -> torch.Tensor:
        """PyTorchベースの簡易コンプレッサー（バッチ非対応・モノラル想定）"""
        # RMSで音量を測定し、しきい値超過分を圧縮
        eps = 1e-8
        rms = torch.sqrt(torch.mean(audio ** 2) + eps)
        threshold = 10 ** (threshold_db / 20)
        over_threshold = torch.abs(audio) > threshold
        compressed = audio.clone()
        compressed[over_threshold] = torch.sign(audio[over_threshold]) * (
            threshold + (torch.abs(audio[over_threshold]) - threshold) / ratio
        )
        return compressed


    def _pytorch_gain(self, audio: torch.Tensor, gain_db: float = 0.0) -> torch.Tensor:
        """PyTorchベースのゲイン調整"""
        gain = 10 ** (gain_db / 20)
        return audio * gain


    def process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """バッチ処理対応"""
        results = []
        for i in range(batch.size(0)):
            results.append(self.enhance(batch[i]))
        return torch.stack(results)