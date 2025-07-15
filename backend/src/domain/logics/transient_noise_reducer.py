import librosa
import pickle
import torch
import torch.nn as nn
import noisereduce as nr
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from df import enhance, init_df
from torchaudio.transforms import Resample

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
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in the range [0.0, 1.0], but got {overlap}")
        self.hop_length = int(frame_size * (1 - overlap))
        self.use_hybrid = use_hybrid
        self.energy_history = deque(maxlen=100)  # エネルギー履歴バッファ
        self.noise_floor = None  # ノイズフロアの初期化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = 48000  # DeepFilterNetの要件サンプルレート

        # リサンプリング設定
        self.resampler = Resample(
            orig_freq=self.sr,
            new_freq=self.target_sr,
            resampling_method='sinc_interp_kaiser'
        )
        self.inverse_resampler = Resample(
            orig_freq=self.target_sr,
            new_freq=self.sr,
            resampling_method='sinc_interp_kaiser'
        )

        # RNNモデルの初期化
        self.rnn_model, self.df_state, suffix = init_df(model_base_dir=rnn_model_path) if rnn_model_path else (None, None, None)

        # ノイズタイプ別処理マップ
        self.method_map = {
            "wind": self._rnn_model_inference,
            "transient": self._rnn_model_inference,
            "other": self._rnnoise_and_binary_mask
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

            # shapeの確認と変換
            for i, frame in enumerate(processed_frames):
                if not isinstance(frame, torch.Tensor):
                    logger.error(f"フレーム {i} の処理結果がTensorではありません: {type(frame)}")
                    processed_frames[i] = frames[i]
                elif frame.dim() == 1:
                    logger.warning(f"フレーム {i} のshapeが1次元です。2次元に変換します: {frame.shape}")
                    processed_frames[i] = frame.unsqueeze(0)
                elif frame.dim() != 2 or frame.size(0) != 1:
                    logger.error(f"フレーム {i} のshapeが不正です: {frame.shape}")
                    processed_frames[i] = frames[i]

            processed_frames = torch.stack(processed_frames)
            return self._reconstruct_from_frames(processed_frames, original_channels)
        except Exception as e:
            logger.error(f"ノイズ除去処理エラー@TransientNoiseReducer.reduce: {e}", exc_info=True)
            return waveform


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
    

    def _classify_noise(self, frame: torch.Tensor) -> str:
        """ノイズタイプ分類"""
        frame_np = frame.numpy()
        # スペクトル重心とゼロクロス率を特徴量として使用
        spectral_centroid = librosa.feature.spectral_centroid(y=frame_np, sr=self.sr)
        zcr = librosa.feature.zero_crossing_rate(frame_np)
        
        if self._is_transient_noise(frame):
           return "transient"
        elif spectral_centroid.mean() < 500 and zcr.mean() < 0.1:
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
        if self.use_hybrid and self.rnn_model and noise_type in ["wind", "transient"]:
            rnn_output = self._rnn_model_inference(frame, noise_type)
            rnnoise_output = self._rnnoise_processing(frame, noise_type)
            binary_mask_output = self._binary_mask(frame, noise_type)
            blended = self._blend_outputs(
                original_frame=original_frame,
                rnn_output=rnn_output,
                rnnoise_output=rnnoise_output,
                binary_mask_output=binary_mask_output,
                noise_type=noise_type
            )
            return blended
        
        # 基本処理
        processor = self.method_map.get(noise_type, self._rnnoise_and_binary_mask)
        return processor(frame, noise_type)


    def _blend_outputs(self, 
                       original_frame: torch.Tensor,  # 元のフレーム(信頼度計算用)
                       rnn_output: torch.Tensor, 
                       rnnoise_output: torch.Tensor,
                       binary_mask_output: torch.Tensor,
                       noise_type: str) -> torch.Tensor:
        # 各手法の信頼度計算
        rnn_confidence = self._calculate_confidence(
            original_frame, rnn_output, noise_type
        )
        rnnoise_confidence = self._calculate_confidence(
            original_frame, rnnoise_output, noise_type
        )
        binary_mask_confidence = self._calculate_confidence(
            original_frame, binary_mask_output, noise_type
        )
        
        # 合計で正規化
        total = rnn_confidence + rnnoise_confidence + binary_mask_confidence + 1e-8
        r = rnn_confidence / total
        n = rnnoise_confidence / total
        b = binary_mask_confidence / total

        base_ratios = {
            "wind": (0.7, 0.2, 0.1),        # (RNN, RNNoise, BinaryMask)
            "transient": (0.8, 0.05, 0.15),   # 突発音はバイナリーマスクも重視
            "other": (0.4, 0.4, 0.2)
        }
        base_r, base_n, base_b = base_ratios.get(noise_type, (1/3, 1/3, 1/3))

        # 信頼度とベース重みの加重平均
        final_r = 0.7 * base_r + 0.3 * r
        final_n = 0.7 * base_n + 0.3 * n
        final_b = 0.7 * base_b + 0.3 * b
        total = final_r + final_n + final_b + 1e-8

        logger.debug(f"ノイズタイプ: {noise_type}\n 信頼度: RNN{rnn_confidence:.2f}, RNNoise{rnnoise_confidence:.2f}, binary_mask{binary_mask_confidence:.2f}\n 最終比率: rnn{final_r:.2f}, rnnoise{final_n:.2f}, binary_mask{final_b:.2f}")

        return (final_r / total) * rnn_output + (final_n / total) * rnnoise_output + (final_b / total) * binary_mask_output


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
        n_fft = 1024
        window = torch.hann_window(n_fft, device=original.device)
        orig_spec = torch.abs(torch.stft(
            original,
            n_fft=n_fft,
            window=window,
            hop_length=n_fft // 4,
            win_length=n_fft,
            center=True,
            return_complex=True
        ))
        proc_spec = torch.abs(torch.stft(
            processed,
            n_fft=n_fft,
            window=window,
            hop_length=n_fft // 4,
            win_length=n_fft,
            center=True,
            return_complex=True
        ))
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
        スケーリングパラメータ2は、スコア0.5を中心に急峻な変化を持たせるために経験的に選択。
        （値を大きくすると0.5付近で急激に0/1へ遷移、小さくすると滑らかになる）
        """
        return 1 / (1 + np.exp(-2 * (score - 0.5)))
    

    def _rnn_model_inference(self, frame: torch.Tensor, noise_type: str) -> torch.Tensor:
        """
        SpeechBrain SepFormerによるノイズ除去
        :param frame: [1, samples] (float32, [-1, 1])
        :return: ノイズ除去済み波形（torch.Tensor, shape: [1, samples]）
        """
        if self.rnn_model is None or self.df_state is None:
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
            if self.sr != self.target_sr:
                frame = self.resampler(frame)  # リサンプリング

            # 2. 推論実行
            denoised = enhance(self.rnn_model, self.df_state, frame)

            # 3 出力値の異常チェック（±1.0を大きく超える値があれば警告）
            if torch.max(torch.abs(denoised)) > 5.0:
                logger.warning(f"RNN推論の出力値が異常に大きい値を含みます (max={torch.max(torch.abs(denoised)).item():.2f})。推論前のデータを返します。")
                return frame
            # 3.5. 出力のNaN/infチェック
            if torch.isnan(denoised).any() or torch.isinf(denoised).any():
                logger.error("RNN出力にNaNまたはinfが含まれています")
                return frame
            
            # 4. 元のサンプルレートに変換
            if self.sr != self.target_sr:
                denoised = self.inverse_resampler(denoised)
            
            if not self.use_hybrid:
                logger.debug(f"RNNモデル推論成功: ノイズタイプ={noise_type}, フレームサイズ={frame.size(1)}, 出力サイズ={denoised.size(1)}")
            return denoised
        except Exception as e:
            logger.error(f"RNNモデル推論エラー@TransientNoiseReducer._rnn_model_inference: {e}")
            return frame
        
    
    def _update_noise_floor(self, frame: np.ndarray):
        """ノイズフロアの適応的更新"""
        frame_energy = np.mean(frame**2)
        if np.isnan(frame_energy) or np.isinf(frame_energy):
            logger.warning(f"ノイズフロア推定: frame_energyが異常値です (NaN/inf): {frame_energy}")
            return
        self.energy_history.append(frame_energy)
        
        # ノイズフロアを下位10パーセンタイルで推定
        if len(self.energy_history) > 10:
            noise_floor = np.percentile(self.energy_history, 10)
            if np.isnan(noise_floor) or np.isinf(noise_floor):
                logger.warning(f"ノイズフロア推定: noise_floorが異常値です (NaN/inf): {noise_floor}")
            else:
                self.noise_floor = noise_floor 


    def _estimate_snr(self, frame: np.ndarray) -> float:
        """SNR推定（dB単位）"""
        signal_energy = np.mean(frame**2)
        if np.isnan(signal_energy) or np.isinf(signal_energy):
            logger.warning(f"SNR推定: signal_energyが異常値です (NaN/inf): {signal_energy}")
            return 30.0  # デフォルト値

        if self.noise_floor is None or self.noise_floor < 1e-8:
            return 30.0  # デフォルト値
        
        snr_db = 10 * np.log10(signal_energy / self.noise_floor)
        if np.isnan(snr_db) or np.isinf(snr_db):
            logger.warning(f"SNR推定: snr_dbが異常値です (NaN/inf): {snr_db} (signal_energy={signal_energy}, noise_floor={self.noise_floor})")
            return 30.0  # デフォルト値

        return max(min(snr_db, 30), -10)  # -10～30dBにクリップ
    

    def _adjust_prop_decrease(self, snr_db: float, noise_type: str) -> float:
        """
        SNRとノイズタイプに基づくprop_decrease調整
        :param snr_db: 推定SNR（dB単位）
        :param noise_type: ノイズタイプ（'wind', 'transient', 'other'）
        :return: 調整後のprop_decrease値（0.3~0.9）
        """
        # prop_decreaseの標準値を0.7とし、SNRが高いほど値が下がるように調整
        # SNR=10dBで0.7、SNR=30dBで0.5、SNR=-10dBで0.9程度になるように線形マッピング
        base_prop = 0.7 - ((snr_db - 10) / 40) * 0.4

        # ノイズタイプ別補正
        correction_factors = {
            "wind": 1.0,       # 風ノイズは標準
            "transient": 0.85, # 瞬発ノイズはやや抑制
            "other": 0.9       # その他はさらに抑制
        }
        prop = base_prop * correction_factors.get(noise_type, 0.9)
        # prop_decreaseは0.3～0.9の範囲にクリップ
        prop = max(0.3, min(prop, 0.9))
        return prop
    

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


    def _binary_mask(self, frame: torch.Tensor, noise_type: str) -> torch.Tensor:
        """
        モノラル2次元テンソル [1, samples] または [samples] に対応。
        入力 shape: [1, frame_size] または [frame_size] のみサポート。
        """
        try:
            # [samples] → [1, samples] に変換（モノラルのみ対応）
            if frame.dim() == 1:
                frame = frame.unsqueeze(0)
            if frame.dim() != 2 or frame.size(0) != 1:
                logger.warning("バイナリマスク処理はモノラル2次元テンソル（[1, samples]）のみ対応しています。入力shape: %s", frame.shape)
                raise NotImplementedError("複数チャンネルやバッチ処理には未対応です。モノラル2次元テンソルのみサポートします。")
            frame_np = frame.squeeze(0).numpy()  # [1, N] → [N]
            n_fft = 1024
            hop_length = n_fft // 4
            win_length = n_fft
            stft = librosa.stft(frame_np, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)
            mag = np.abs(stft)
            
            # 適応的閾値設定
            threshold = self._calculate_adaptive_threshold(mag)
            # thresholdが1次元なら2次元にreshapeしてブロードキャスト
            if threshold.ndim == 1 and mag.ndim == 2:
                threshold = threshold[:, np.newaxis]
            mask = mag > threshold

            # 複素スペクトルにマスク適用
            clean_stft = stft * mask.astype(stft.dtype)
            # stftのcenter引数と合わせる
            clean_frame = librosa.istft(clean_stft, hop_length=hop_length, win_length=win_length, center=True)
            # clean_frameの長さをframeに合わせて厳密に調整
            target_len = frame_np.shape[-1]
            if len(clean_frame) > target_len:
                clean_frame = clean_frame[:target_len]
            elif len(clean_frame) < target_len:
                clean_frame = np.pad(clean_frame, (0, target_len - len(clean_frame)))
            return torch.from_numpy(clean_frame).unsqueeze(0)  # [N] → [1, N]
        except Exception as e:
            logger.error(f"バイナリマスク処理エラー@TransientNoiseReducer._binary_mask: {e}", exc_info=True)
            return frame
    

    def _calculate_adaptive_threshold(self, mag: np.ndarray) -> np.ndarray:
        """
        適応的閾値計算
        :param mag: STFT振幅スペクトル（shape: [freq_bins] または [freq_bins, frames] に対応）。
            1次元（[freq_bins]）の場合は1フレーム分、2次元（[freq_bins, frames]）の場合は複数フレーム分のスペクトルを想定。
            2次元の場合は各フレームごと（axis=1）に処理されます。
        :return: 各周波数ビンごとの閾値（shape: [freq_bins] または [freq_bins, frames]）
        """
        # mag: [freq_bins] または [freq_bins, frames] に対応
        window_size = 5  # 例: 5点移動平均
        half_win = window_size // 2
        local_energy = np.zeros_like(mag)
        if mag.ndim == 1:
            for i in range(len(mag)):
                start = max(0, i - half_win)
                end = min(len(mag), i + half_win + 1)
                local_energy[i] = np.mean(mag[start:end])
            noise_floor = np.percentile(mag, 30)
        elif mag.ndim == 2:
            for f in range(mag.shape[1]):
                for i in range(mag.shape[0]):
                    start = max(0, i - half_win)
                    end = min(mag.shape[0], i + half_win + 1)
                    local_energy[i, f] = np.mean(mag[start:end, f])
            noise_floor = np.percentile(mag, 30, axis=0)
        else:
            raise ValueError("mag must be 1D or 2D array")
        threshold = noise_floor + 0.2 * local_energy
        return threshold
    

    def _rnnoise_and_binary_mask(self, frame: torch.Tensor, noise_type: str) -> torch.Tensor:
        """
        RNNoiseとバイナリーマスクのハイブリッド処理
        """
        rnnoise_out = self._rnnoise_processing(frame, noise_type)
        binary_mask_out = self._binary_mask(frame, noise_type)
        # 1. 信頼度計算
        rn_conf = self._calculate_confidence(frame, rnnoise_out, noise_type)
        bm_conf = self._calculate_confidence(frame, binary_mask_out, noise_type)
        
        # 2. 信頼度差に基づく動的比率調整
        confidence_diff = rn_conf - bm_conf
        dynamic_ratio = 0.5 + 0.5 * torch.sigmoid(torch.tensor(5.0 * confidence_diff, dtype=torch.float32))
        
        # 3. ノイズタイプ別ベース比率と統合
        base_ratios = {
            "wind": 0.8,    # RNNoise重視
            "transient": 0.2, # バイナリーマスク重視
            "other": 0.5
        }
        base_ratio = base_ratios.get(noise_type, 0.5)
        final_ratio = 0.7 * base_ratio + 0.3 * dynamic_ratio.item()

        logger.debug(f"ノイズタイプ: {noise_type}, RNNoise信頼度: {rn_conf:.2f}, バイナリーマスク信頼度: {bm_conf:.2f}, 最終比率: {final_ratio:.2f}")
        # 4. ブレンド実行
        return final_ratio * rnnoise_out + (1 - final_ratio) * binary_mask_out

