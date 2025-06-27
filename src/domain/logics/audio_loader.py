from __future__ import annotations

import torch
import torchaudio
import noisereduce as nr
import numpy as np
from dataclasses import dataclass, field


class AudioLoader:
    @staticmethod
    def load(path: str) -> RawAudio:
        waveform, sample_rate = torchaudio.load(path)
        
        # サンプリングレートを16kHzに変換
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)
            sample_rate = 16000
        
        # モノラル変換（複数チャンネルの場合は平均化）
        if waveform.shape[0] > 1:
            # 複数チャンネルの場合は平均化してモノラルに（2次元を保持）
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # 既にモノラルの場合はそのまま（2次元を保持）
        
        # 念のため2次元であることを確認
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (time,) → (1, time)
        
        return RawAudio.new(
            waveform=waveform,
            sample_rate=sample_rate,
            path=path
        )

@dataclass(frozen=False, eq=True)
class RawAudio:
    """音声ファイルの生データをカプセル化するエンティティ"""
    _waveform: torch.Tensor
    _sample_rate: int
    _path: str
    _duration: float = field(init=False)
                             
    
    def __post_init__(self):
        # 波形の長さから音声の長さを計算
        self._duration = self._waveform.size(1) / self._sample_rate


    @classmethod
    def new(cls, waveform: torch.Tensor, sample_rate: int, path: str) -> RawAudio:
        """
        新しいRawAudioを作成
        :param waveform: 音声波形データ（2次元Tensor）
        :param sample_rate: サンプリングレート
        :param path: 音声ファイルのパス
        :return: RawAudioインスタンス
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (time,) → (1, time) の2次元に変換
        elif waveform.dim() != 2:
            raise ValueError("Waveform must be a 1D or 2D tensor. Received: {}".format(waveform.dim())) 
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer. Received: {}".format(sample_rate))
        if not isinstance(path, str) or not path:
            raise ValueError("Path must be a non-empty string. Received: {}".format(path))
        if waveform.shape[0] != 1:
            raise ValueError("Waveform must be mono (1 channel). Received: {}".format(waveform.shape[0]))
        
        return cls(
            _waveform=waveform,
            _sample_rate=sample_rate,
            _path=path
        )
    

    @property
    def waveform(self):
        """波形データを取得"""
        return self._waveform

    @property
    def sample_rate(self):
        """サンプリングレートを取得"""
        return self._sample_rate

    @property
    def path(self):
        """音声ファイルのパスを取得"""
        return self._path

    @property
    def duration(self):
        """音声の長さ（秒）を取得"""
        return self._duration