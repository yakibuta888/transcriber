import librosa
import pickle
import torch
import torch.nn as nn
import noisereduce as nr
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from torchaudio.transforms import Resample

from models.CleanUNet.network import CleanUNet
from settings import logger


class TransientNoiseReducer:
    def __init__(self, 
                 sample_rate=16000,
                 rnn_model_path=None,
                 use_hybrid=True,
                 frame_size=2048,
                 overlap=0.5,
                 quantize=True):
        """
        非定常ノイズ除去の最適化実装
        
        :param rnn_model_path: カスタムRNNモデルのパス
        :param use_hybrid: ハイブリッド処理の有効化
        :param frame_size: 処理フレームサイズ
        :param overlap: フレームオーバーラップ率
        :param quantize: 量子化の有無
        """
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_length = int(frame_size * (1 - overlap))
        self.use_hybrid = use_hybrid
        self.energy_history = deque(maxlen=100)  # エネルギー履歴バッファ
        self.noise_floor = None  # ノイズフロアの初期化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = 48000 # CleanUNetの要件

        # リサンプラー初期化
        self.resampler = Resample(self.sr, self.target_sr)
        self.inverse_resampler = Resample(self.target_sr, self.sr)

        # RNNモデルのロード&量子化
        self.rnn_model = self._load_rnn_model(rnn_model_path, quantize) if rnn_model_path else None
        
        # ノイズタイプ別処理マップ
        self.method_map = {
            "wind": self._rnn_model_inference,
            "transient": self._rnnoise_processing,
            "other": self._binary_mask
        }

    
    def _load_rnn_model(self, model_path: str, quantize: bool) -> nn.Module | None:
        """CleanUNetのRNNモデルをロード"""
        try:
            # モデルアーキテクチャの初期化
            model = CleanUNet()
            
            # .pklファイルから重みをロード
            with open(model_path, 'rb') as f:
                state_dict = pickle.load(f)
            
            # 互換性のあるキー名に変換
            converted_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # 'module.'プレフィックスを削除
                converted_state_dict[k] = v
            
            model.load_state_dict(converted_state_dict)
            model.eval()

            if quantize:
                model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Conv1d, nn.Linear},
                    dtype=torch.qint8
                )

            return model
        except Exception as e:
            logger.error(f"RNNモデルロードエラー: {e}")
            return None
    

    def reduce(self, waveform: torch.Tensor) -> torch.Tensor:
        """ノイズタイプに基づく動的処理（並列化）"""
        try:
            frames = self._segment_into_frames(waveform)
            processed_frames = []

            def process_frame(frame):
                try:
                    noise_type = self._classify_noise(frame)
                    return self._process_by_type(frame, noise_type)
                except Exception as e:
                    logger.error(f"フレーム処理エラー@TransientNoiseReducer.reduce.process_frame: {e}")
                    return frame  # エラー発生時は元のフレームを返す

            with ThreadPoolExecutor() as executor:
                processed_frames = list(executor.map(process_frame, frames))

            processed_frames = torch.stack(processed_frames)
            return self._reconstruct_from_frames(processed_frames)
        except Exception as e:
            logger.error(f"ノイズ除去処理エラー@TransientNoiseReducer.reduce: {e}")
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
        if self.use_hybrid and self.rnn_model:
            rnn_output = self._rnn_model_inference(frame, noise_type)
            rnnoise_output = self._rnnoise_processing(frame, noise_type)
            blended = self._blend_outputs(
                original_frame=original_frame,
                rnn_output=rnn_output,
                rnnoise_output=rnnoise_output,
                noise_type=noise_type
            )
            return blended
        
        # 基本処理
        processor = self.method_map.get(noise_type, self._rnnoise_processing)
        return processor(frame)


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
        spectral_corr = np.corrcoef(orig_spec.flatten().cpu(), proc_spec.flatten().cpu())[0, 1]

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
        """CleanUNetを利用したRNN推論"""
        if self.rnn_model is None:
            # RNNモデルが利用できない場合はRNNoise処理にフォールバック
            return self._rnnoise_processing(frame, noise_type)

        try:
            # 1. 値の範囲制限
            frame = torch.clamp(frame, -1.0, 1.0)
            
            # 2. サンプルレート変換
            audio_48k = self.resampler(frame)

            # 3. 推論実行
            with torch.no_grad():
                denoised_48k = self.rnn_model(audio_48k.unsqueeze(0).to(self.device))
            
            # 4. 元のサンプルレートに変換
            return self.inverse_resampler(denoised_48k.squeeze(0).cpu())
        except Exception as e:
            logger.error(f"RNNモデル推論エラー@TransientNoiseReducer._rnn_model_inference: {e}")
            return frame
        
    
    def _update_noise_floor(self, frame: np.ndarray):
        """ノイズフロアの適応的更新（検索結果[1]に基づく）"""
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
        （検索結果[1][2]のアルゴリズム拡張）
        """
        # 基本調整（検索結果[1]）
        base_prop = 1.0 - (snr_db + 10) / 40 * 0.7
        
        # ノイズタイプ別補正（検索結果[2][6]）
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
            
            # RNNoise処理（検索結果[2]）
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
    

    def _segment_into_frames(self, waveform: torch.Tensor) -> torch.Tensor:
        """波形をフレーム分割（オーバーラップ付き、ベクトル化）"""
        # shape: (num_frames, frame_size)
        frames = waveform.unfold(0, self.frame_size, self.hop_length)
        return frames


    def _reconstruct_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """フレームを再結合（オーバーラップ加算、ベクトル化）"""
        num_frames = frames.size(0)
        output_len = self.frame_size + self.hop_length * (num_frames - 1)
        output = torch.zeros(output_len, device=frames.device)
        window = torch.hann_window(self.frame_size, device=frames.device)
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_size
            output[start:end] += frames[i] * window
        return output


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


    def _binary_mask(self, frame: torch.Tensor) -> torch.Tensor:
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
            logger.error(f"バイナリマスク処理エラー@TransientNoiseReducer._binary_mask: {e}")
            return frame
    

    def _calculate_adaptive_threshold(self, mag: np.ndarray) -> float:
        """適応的閾値計算"""
        # 周辺フレームのエネルギーを考慮
        local_energy = np.mean([mag[max(0, i-2):i+2] for i in range(mag.shape[0])])

        # ノイズフロア推定
        noise_floor = np.percentile(mag, 30)

        return (noise_floor + 0.2 * local_energy).item()
    