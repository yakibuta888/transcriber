import torch
import torchaudio


class AudioLoader:
    def __init__(self, path):
        self._path = path
        waveform, sample_rate = torchaudio.load(path)
        # サンプリングレートを16kHzに変換
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)
            sample_rate = 16000
        # モノラル変換（複数チャンネルの場合は平均化）
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        self._waveform = waveform
        self._sample_rate = sample_rate
    

    @property
    def path(self):
        """音声ファイルのパスを取得"""
        return self._path

        
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
        return self._waveform.squeeze().numpy(), self._sample_rate

    def for_whisper(self):
        """whisperパイプライン用（辞書形式）"""
        array = self._waveform.numpy()
        return {"array": array, "sampling_rate": self._sample_rate}
    
    def for_pyannote(self):
        """pyannoteパイプライン用（辞書形式）"""
        array = self._waveform.numpy()
        return {"waveform": array, "sample_rate": self._sample_rate}