import librosa
import pickle
import torch
import torch.nn as nn
import noisereduce as nr
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from speechbrain.inference.separation import SepformerSeparation
from torchaudio.transforms import Resample

from models.CleanUNet.network import CleanUNet
from settings import logger


class TransientNoiseReducer:
    def __init__(
        self, 
        sample_rate=16000,
        rnn_model_path=None,
        use_hybrid=True,
        frame_size=2048,
        overlap=0.5,
    ):
        """
        非定常ノイズ除去の最適化実装
        
        :param rnn_model_path: カスタムRNNモデルのパス
        :param use_hybrid: ハイブリッド処理の有効化
        :param frame_size: 処理フレームサイズ
        :param overlap: フレームオーバーラップ率
        """
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_length = int(frame_size * (1 - overlap))
        self.use_hybrid = use_hybrid
        self.energy_history = deque(maxlen=100)  # エネルギー履歴バッファ
        self.noise_floor = None  # ノイズフロアの初期化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RNNモデルのロード&量子化
        self.rnn_model = SepformerSeparation.from_hparams(
            source=rnn_model_path,
            run_opts={"device": self.device}
        )

        # ノイズタイプ別処理マップ
        self.method_map = {
            "wind": self._rnn_model_inference,
            "transient": self._rnnoise_processing,
            "other": self._binary_mask
        }

    
    def reduce(self, waveform: torch.Tensor) -> torch.Tensor:
        """ノイズタイプに基づく動的処理（並列化）"""
        try:
            original_channels = self.get_num_channels(waveform)
            frames = self._segment_into_frames(waveform)
            processed_frames = []

            def process_frame(frame):
                try:
                    noise_type = self._classify_noise(frame)
                    return self._process_by_type(frame, noise_type)
                except Exception as e:
                    logger.error(f"フレーム処理エラー@TransientNoiseReducer.reduce.process_frame: {e}", exc_info=True)
                    return frame  # エラー発生時は元のフレームを返す

            with ThreadPoolExecutor() as executor:
                processed_frames = list(executor.map(process_frame, frames))

            processed_frames = torch.stack(processed_frames)
            return self._reconstruct_from_frames(processed_frames, original_channels)
        except Exception as e:
            logger.error(f"ノイズ除去処理エラー@TransientNoiseReducer.reduce: {e}", exc_info=True)
            return waveform
    

    def _classify_noise(self, frame: torch.Tensor) -> str:
        """ノイズタイプ分類"""
        # スペクトル重心とゼロクロス率を特徴量として使用
        spectral_centroid = librosa.feature.spectral_centroid(y=frame.numpy(), sr=self.sr)
        zcr = librosa.feature.zero_crossing_rate(frame.numpy())
        
        if spectral_centroid.mean() < 500 and zcr.mean() < 0.1:
            return "wind"  # 低周波・低ゼロクロス→風切り音
        elif zcr.mean() > 0.25:
            return "transient"  # 高ゼロクロス→瞬発音
        else:
            return "other"


    def _process_by_type(self, frame: torch.Tensor, noise_type: str) -> torch.Tensor:
        # 元のフレームを保持（信頼度計算用）
        original_frame = frame.clone()

        """ノイズタイプに応じた処理選択"""
        # ハイブリッド処理（移動平均法に基づく判定）
        if self.use_hybrid and self.rnn_model and False:
            rnn_output = self._rnn_model_inference(frame, noise_type)
            rnnoise_output = self._rnnoise_processing(frame, noise_type)
            blended = self._blend_outputs(
                original_frame=original_frame,
                rnn_output=rnn_output,
                rnnoise_output=rnnoise_output,
                noise_type=noise_type
            )
            return blended
        
        noise_type = "wind"  # debug
        # 基本処理
        processor = self.method_map.get(noise_type, self._rnnoise_processing)
        return processor(frame, noise_type)


    def _blend_outputs(self, 
                       original_frame: torch.Tensor,  # 元のフレーム(信頼度計算用)
                       rnn_output: torch.Tensor, 
                       rnnoise_output: torch.Tensor,
                       noise_type: str) -> torch.Tensor:
        # 1. 各手法の信頼度計算
        rnn_confidence = self._calculate_confidence(
            original_frame, rnn_output, noise_type
        )
        rnnoise_confidence = self._calculate_confidence(
            original_frame, rnnoise_output, noise_type
        )
        
        # 2. 信頼度差に基づく動的比率調整
        confidence_diff = rnn_confidence - rnnoise_confidence
        dynamic_ratio = 0.5 + 0.5 * torch.sigmoid(torch.tensor(5.0 * confidence_diff))
        
        # 3. ノイズタイプ別ベース比率と統合
        base_ratios = {
            "wind": 0.8,    # RNN重視
            "transient": 0.3, # RNNoise重視
            "other": 0.5
        }
        base_ratio = base_ratios.get(noise_type, 0.5)
        final_ratio = 0.7 * base_ratio + 0.3 * dynamic_ratio.item()
        
        logger.debug(f"ノイズタイプ: {noise_type}, RNN信頼度: {rnn_confidence:.2f}, RNNoise信頼度: {rnnoise_confidence:.2f}, 最終比率: {final_ratio:.2f}")
        # 4. ブレンド実行
        return final_ratio * rnn_output + (1 - final_ratio) * rnnoise_output


    def _calculate_confidence(self, 
                             original_frame: torch.Tensor, 
                             processed_frame: torch.Tensor,
                             noise_type: str) -> float:
        """
        ノイズ除去品質の信頼度を計算（0.0-1.0）
        
        :param original_frame: 元の音声フレーム
        :param processed_frame: 処理済み音声フレーム
        :param noise_type: ノイズタイプ（'wind','transient','other'）
        :return: 信頼度スコア
        """
        # 特徴量抽出
        features = self._extract_confidence_features(original_frame, processed_frame)
        
        # ノイズタイプ別評価指標の重み付け
        weights = self._get_noise_type_weights(noise_type)
        
        # 複合スコア計算
        composite_score = (
            weights['snr_improvement'] * features['snr_improvement'] +
            weights['spectral_corr'] * features['spectral_corr'] +
            weights['residual_flatness'] * features['residual_flatness']
        )
        
        # 統計的正規化
        normalized_score = self._sigmoid_normalization(composite_score)
        
        return normalized_score


    def _extract_confidence_features(self, 
                                    original: torch.Tensor, 
                                    processed: torch.Tensor) -> dict:
        """
        信頼度計算用の音響特徴量を抽出
        """
        orig_np = original.numpy()
        proc_np = processed.numpy()
        
        # SNR改善量
        orig_energy = np.sum(orig_np ** 2)
        residual = orig_np - proc_np
        resid_energy = np.sum(residual ** 2)
        snr_improvement = 10 * np.log10(orig_energy / (resid_energy + 1e-8))

        # スペクトル相関（PyTorchで計算）
        orig_spec = torch.abs(torch.stft(original, n_fft=1024, return_complex=True))
        proc_spec = torch.abs(torch.stft(processed, n_fft=1024, return_complex=True))
        orig_flat = orig_spec.flatten().cpu().numpy()
        proc_flat = proc_spec.flatten().cpu().numpy()
        # 入力値のチェックと例外処理
        if orig_flat.size == 0 or proc_flat.size == 0 or np.all(orig_flat == orig_flat[0]) or np.all(proc_flat == proc_flat[0]):
            spectral_corr = 0.0
        else:
            try:
                spectral_corr = np.corrcoef(orig_flat, proc_flat)[0, 1]
                if np.isnan(spectral_corr):
                    spectral_corr = 0.0
            except Exception:
                spectral_corr = 0.0

        # 残差ノイズの平坦度
        resid_flatness = librosa.feature.spectral_flatness(y=residual)[0].mean()
        
        return {
            'snr_improvement': np.clip(snr_improvement / 20, 0, 1),  # 0-1に正規化
            'spectral_corr': (spectral_corr + 1) / 2,  # -1~1 → 0~1
            'residual_flatness': resid_flatness
        }


    def _get_noise_type_weights(self, noise_type: str) -> dict:
        """
        ノイズタイプ別の特徴量重み付け
        """
        weights = {
            'wind': {'snr_improvement': 0.2, 'spectral_corr': 0.3, 'residual_flatness': 0.5},
            'transient': {'snr_improvement': 0.6, 'spectral_corr': 0.3, 'residual_flatness': 0.1},
            'other': {'snr_improvement': 0.4, 'spectral_corr': 0.4, 'residual_flatness': 0.2}
        }
        return weights.get(noise_type, weights['other'])


    def _sigmoid_normalization(self, score: float) -> float:
        """
        シグモイド関数によるスコア正規化
        """
        return 1 / (1 + np.exp(-2 * (score - 0.5)))
    

    def _rnn_model_inference(self, frame: torch.Tensor, noise_type: str) -> torch.Tensor:
        """
        SpeechBrain SepFormerによるノイズ除去
        :param frame: [1, samples] (float32, [-1, 1])
        :return: ノイズ除去済み波形（torch.Tensor, shape: [1, samples]）
        """
        if self.rnn_model is None:
            # RNNモデルが利用できない場合はRNNoise処理にフォールバック
            logger.warning("RNNモデルがロードされていません。RNNoise処理にフォールバックします。")
            return self._rnnoise_processing(frame, noise_type)

        try:
            # 1. validate input shape
            if frame.dim() != 1 and frame.dim() != 2:
                raise ValueError(f"Unexpected frame shape: {frame.shape}. Expected 1D or 2D tensor.")
            # [samples] -> [1, samples]
            if frame.dim() == 1:
                frame = frame.unsqueeze(0) 
            # 1.5 チャンネル数のチェック
            if frame.dim() == 2 and frame.size(0) != 1:
                raise ValueError(f"Unexpected frame channels: {frame.size(0)}. Expected 1 channel (mono).")
            # 1.6 サンプルレートのチェック
            if self.sr != 16000:
                raise ValueError(f"Unexpected sample rate: {self.sr}. Expected 16000 Hz for CleanUNet.")

            # 2. 推論実行
            with torch.no_grad():
                # 出力: [1, samples, n_sources]（通常n_sources=2）
                est_sources = self.rnn_model.separate_batch(frame)
                # 通常は最初のソースを返す（話者分離でなくノイズ除去用途なら）
                denoised = est_sources[:, :, 0]

            # 3 出力値の異常チェック（±1.0を大きく超える値があれば警告）
            if torch.max(torch.abs(denoised)) > 5.0:
                logger.warning(f"RNN推論の出力値が異常に大きい値を含みます (max={torch.max(torch.abs(denoised)).item():.2f})。推論前のデータを返します。")
                # return frame
                # FIXME: ここでの処理は、異常値を含む場合に元のフレームを返すようにする
            # 3.5. 出力のNaN/infチェック
            if torch.isnan(denoised).any() or torch.isinf(denoised).any():
                logger.error("RNN出力にNaNまたはinfが含まれています")
                return frame
            return denoised
        except Exception as e:
            logger.error(f"RNNモデル推論エラー@TransientNoiseReducer._rnn_model_inference: {e}")
            return frame
        
    
    def _update_noise_floor(self, frame: np.ndarray):
        """ノイズフロアの適応的更新"""
        frame_energy = np.mean(frame**2)
        self.energy_history.append(frame_energy)
        
        # ノイズフロアを下位10パーセンタイルで推定
        if len(self.energy_history) > 10:
            self.noise_floor = np.percentile(self.energy_history, 10)
    

    def _estimate_snr(self, frame: np.ndarray) -> float:
        """SNR推定（dB単位）"""
        signal_energy = np.mean(frame**2)
        if self.noise_floor is None or self.noise_floor < 1e-8:
            return 30.0  # デフォルト値
        
        snr_db = 10 * np.log10(signal_energy / self.noise_floor)
        return max(min(snr_db, 30), -10)  # -10～30dBにクリップ
    

    def _adjust_prop_decrease(self, snr_db: float, noise_type: str) -> float:
        """
        SNRとノイズタイプに基づくprop_decrease調整
        :param snr_db: 推定SNR（dB単位）
        :param noise_type: ノイズタイプ（'wind', 'transient', 'other'）
        :return: 調整後のprop_decrease値（0.0-1.0）
        """
        # 基本調整
        base_prop = 1.0 - (snr_db + 10) / 40 * 0.7

        # ノイズタイプ別補正
        correction_factors = {
            "wind": 0.9,    # 風ノイズは積極的除去
            "transient": 0.7,
            "other": 1.0
        }
        return base_prop * correction_factors.get(noise_type, 1.0)
    

    def _rnnoise_processing(self, frame: torch.Tensor, noise_type: str) -> torch.Tensor:
        """調整済みprop_decreaseでRNNoiseを適用"""
        try:
            frame_np = frame.numpy()
            
            # SNR推定とprop_decrease計算
            self._update_noise_floor(frame_np)
            snr_db = self._estimate_snr(frame_np)
            prop_decrease = self._adjust_prop_decrease(snr_db, noise_type)
            
            # RNNoise処理
            reduced = nr.reduce_noise(
                y=frame_np,
                sr=self.sr,
                stationary=False,
                prop_decrease=prop_decrease,
                time_constant_s=0.5  # 高速応答設定
            )
            return torch.from_numpy(reduced)
        except Exception as e:
            logger.error(f"RNNoise処理エラー@TransientNoiseReducer._rnnoise_processing: {e}")
            return frame
    
    
    def get_num_channels(self, waveform: torch.Tensor) -> int:
        """波形テンソルからチャンネル数を取得"""
        if waveform.dim() == 1:
            return 1
        elif waveform.dim() == 2:
            return waveform.size(0)
        else:
            raise ValueError(f"サポートされない波形の次元数: {waveform.dim()}")

    
    def _segment_into_frames(self, waveform: torch.Tensor) -> torch.Tensor:
        """波形をフレーム分割（オーバーラップ付き、ベクトル化）"""
        # 入力形状を正規化: [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples] 
        channels, total_samples = waveform.shape

        # パディング（端のフレームが足りない場合）
        pad_len = (self.hop_length - (total_samples - self.frame_size) % self.hop_length) % self.hop_length
        if pad_len > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # ウィンドウ関数
        window = torch.hann_window(self.frame_size, device=waveform.device)

        # 時間次元（次元1）で展開
        # 結果の形状: [channels, num_frames, frame_size]
        frames = waveform.unfold(
            dimension=1,
            size=self.frame_size,
            step=self.hop_length
        )
        frames = frames * window  # ウィンドウ適用
        
        # フレームとチャンネルを統合: [num_frames * channels, frame_size]
        return frames.permute(1, 0, 2).reshape(-1, self.frame_size)


    def _reconstruct_from_frames(self, frames: torch.Tensor, original_channels: int) -> torch.Tensor:
        """フレームを再結合（オーバーラップ加算）"""
        # フレームを元の構造に戻す: [num_frames * channels, frame_size] -> [channels, num_frames, frame_size]
        num_frames = frames.size(0) // original_channels
        frames = frames.view(num_frames, original_channels, self.frame_size).permute(1, 0, 2)
        
        # 出力バッファの初期化
        output_len = self.frame_size + self.hop_length * (num_frames - 1)
        output = torch.zeros(original_channels, output_len, device=frames.device)
        
        # ハニングウィンドウを準備
        window = torch.hann_window(self.frame_size, device=frames.device)
        
        # オーバーラップ加算処理
        window_sum = torch.zeros(output_len, device=frames.device)
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_size
            output[:, start:end] += frames[:, i, :] * window
            window_sum[start:end] += window

        # ゼロ割防止
        window_sum = torch.clamp(window_sum, min=1e-8)
        output = output / window_sum  # 正規化

        pad_remove = self.frame_size - self.hop_length
        if pad_remove > 0:
            return output[..., :-pad_remove]  # パディング分を除去
        else:
            return output  # パディング不要


    def _is_transient_noise(self, frame: torch.Tensor) -> bool:
        """
        フレーム内に非定常ノイズが存在するか判定
        移動平均法に基づく判定
        """
        # ゼロクロス率と振幅エンベロープの急変を検出
        diff = torch.diff(torch.abs(frame))
        max_diff = torch.max(diff).item()
        zero_crossings = torch.sum(torch.diff(frame > 0)).item() / len(frame)
        
        # ノイズ判定基準（経験的閾値）
        return max_diff > 0.3 or zero_crossings > 0.25


    def _binary_mask(self, frame: torch.Tensor, noise_type: str | None = None) -> torch.Tensor:
        try:
            frame_np = frame.numpy()
            stft = librosa.stft(frame_np, n_fft=1024)
            mag = np.abs(stft)
            
            # 適応的閾値設定
            threshold = self._calculate_adaptive_threshold(mag)
            mask = mag > threshold
            
            # 複素スペクトルにマスク適用
            clean_stft = stft * mask
            clean_frame = librosa.istft(clean_stft)
            return torch.from_numpy(clean_frame)
        except Exception as e:
            logger.error(f"バイナリマスク処理エラー@TransientNoiseReducer._binary_mask: {e}", exc_info=True)
            return frame
    

    def _calculate_adaptive_threshold(self, mag: np.ndarray) -> np.ndarray:
        """
        適応的閾値計算
        :param mag: STFT振幅スペクトル（shape: [freq_bins, frames] または [freq_bins]）
        :return: 各周波数ビンごとの閾値（shape: [freq_bins]）
        """
        # mag: [freq_bins] または [freq_bins, frames]
        # ここでは1フレーム分（[freq_bins,]）を想定
        window_size = 5  # 例: 5点移動平均
        half_win = window_size // 2
        local_energy = np.zeros_like(mag)
        for i in range(len(mag)):
            start = max(0, i - half_win)
            end = min(len(mag), i + half_win + 1)
            local_energy[i] = np.mean(mag[start:end])
        
        # ノイズフロア推定
        noise_floor = np.percentile(mag, 30)
        # 各ビンごとに閾値を計算
        threshold = noise_floor + 0.2 * local_energy
        return threshold