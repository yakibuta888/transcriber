import torch


class AudioNormalizer:
    @staticmethod
    def peak_normalize(waveform: torch.Tensor) -> torch.Tensor:
        peak = waveform.abs().max()
        if peak > 1e-6:  # ゼロ除算防止
            waveform = waveform / peak
        return waveform

    @staticmethod
    def loudness_normalize(waveform: torch.Tensor) -> torch.Tensor:
        # EBU R128規格に基づくラウドネス正規化
        loudness = waveform.mean(dim=-1, keepdim=True)
        target_loudness = -23.0  # ターゲットラウドネス（dBFS）
        gain = 10 ** (target_loudness / 20) / loudness.abs().max()
        return waveform * gain