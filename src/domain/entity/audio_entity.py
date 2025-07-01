from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class AudioEntity:
    """音声処理結果をカプセル化するエンティティ"""
    _waveform: torch.Tensor  # 処理済み波形データ
    _sample_rate: int        # サンプリングレート
    _duration: float         # 音声の長さ（秒）
    _num_channels: int       # チャンネル数
    _processing_steps: list  # 適用された処理ステップ
    _metadata: dict          # 追加メタデータ
    

    @staticmethod
    def _validate(waveform: torch.Tensor, sample_rate: int, duration: float,
                 num_channels: int, processing_steps: list, metadata: dict):
        # 波形データの次元を確認
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (time,) → (1, time) の2次元に変換
        elif waveform.dim() != 2:
            raise ValueError("Waveform must be a 1D or 2D tensor. Received: {}".format(waveform.dim()))
        actual_channels = waveform.size(0) if waveform.dim() > 1 else 1
        if actual_channels != num_channels:
            raise ValueError("Waveform channels do not match num_channels. Expected: {}, Got: {}".format(num_channels, actual_channels))
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer. Received: {}".format(sample_rate))
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("Duration must be a non-negative number. Received: {}".format(duration))
        if not isinstance(processing_steps, list):
            raise ValueError("Processing steps must be a list. Received: {}".format(type(processing_steps)))
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary. Received: {}".format(type(metadata)))
        

    @classmethod
    def new(cls, waveform: torch.Tensor, sample_rate: int, duration: float,
             num_channels: int, processing_steps: list = [],
             metadata: dict = {}) -> AudioEntity:
        """
        新しいAudioEntityを作成
        :param waveform: 処理済み波形データ（2次元Tensor）
        :param sample_rate: サンプリングレート
        :param duration: 音声の長さ（秒）
        :param num_channels: チャンネル数
        :param processing_steps: 適用された処理ステップのリスト
        :param metadata: 追加メタデータ
        :return: AudioEntityインスタンス
        """

        cls._validate(waveform, sample_rate, duration, num_channels, processing_steps, metadata)

        return cls(
            _waveform=waveform,
            _sample_rate=sample_rate,
            _duration=duration,
            _num_channels=num_channels,
            _processing_steps=processing_steps,
            _metadata=metadata
        )

        
    @property
    def waveform(self):
        """波形データを取得"""
        return self._waveform
        

    @property
    def sample_rate(self):
        """サンプリングレートを取得"""
        return self._sample_rate


    def to_numpy(self):
        """numpy配列とサンプリングレートを返す（NumPyベースの音響処理ライブラリで追加処理を行うとき等に利用）"""
        waveform_cpu = self._waveform.cpu()
        return waveform_cpu.squeeze().numpy(), self._sample_rate


    def for_whisper(self):
        """whisperパイプライン用（1次元numpy配列）"""
        array = self._waveform.cpu().squeeze().numpy()  # (time,)の1次元
        return {"array": array, "sampling_rate": self._sample_rate}
    

    def for_pyannote(self):
        """pyannote.audio用（2次元torch.Tensor）"""
        return {
            "waveform": self._waveform,  # (1, time)の2次元を保持
            "sample_rate": self._sample_rate
        }
