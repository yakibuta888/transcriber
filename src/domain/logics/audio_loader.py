import torchaudio


class AudioLoader:
    def __init__(self, path):
        self.path = path
        self.waveform, self.sample_rate = torchaudio.load(path)
        if self.sample_rate != 16000:
            self.waveform = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate, new_freq=16000
            )(self.waveform)
            self.sample_rate = 16000