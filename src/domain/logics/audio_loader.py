import torch
import torchaudio

from domain.common.progress_reporter import ProgressReporter


class AudioLoader:
    def __init__(self, path, progress: ProgressReporter | None = None):
        self._path = path
        self._steps = 3  # 音声処理のステップ数（進捗表示用）
        
        if progress:
            progress.update.preprocessing(0, "音声ファイルを読み込み中...")
        waveform, sample_rate = torchaudio.load(path)
        
        if progress:
            progress.update.preprocessing(1, "リサンプリング中...")
        # サンプリングレートを16kHzに変換
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(waveform)
            sample_rate = 16000
        
        if progress:
            progress.update.preprocessing(2, "モノラル変換中...")
        # モノラル変換（複数チャンネルの場合は平均化）
        if waveform.shape[0] > 1:
            # 複数チャンネルの場合は平均化してモノラルに（2次元を保持）
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # 既にモノラルの場合はそのまま（2次元を保持）
        
        # 念のため2次元であることを確認
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (time,) → (1, time)
        
        self._waveform = waveform
        self._sample_rate = sample_rate
        
        if progress:
            progress.update.preprocessing(3, "音声ファイルの読み込みが完了しました。")
    

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
    

    @property
    def steps(self):
        """音声処理のステップ数を返す（進捗表示用）"""
        return self._steps


    def to_numpy(self):
        """numpy配列とサンプリングレートを返す（NumPyベースの音響処理ライブラリで追加処理を行うとき等に利用）"""
        return self._waveform.squeeze().numpy(), self._sample_rate

    def for_whisper(self):
        """whisperパイプライン用（1次元numpy配列）"""
        array = self._waveform.squeeze().numpy()  # (time,)の1次元
        return {"array": array, "sampling_rate": self._sample_rate}
    
    def for_pyannote(self):
        """pyannote.audio用（2次元torch.Tensor）"""
        return {
            "waveform": self._waveform,  # (1, time)の2次元を保持
            "sample_rate": self._sample_rate
        }