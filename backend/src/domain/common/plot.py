import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_audio_analysis(waveform, sr, filename):
    plt.figure(figsize=(15, 10))
    
    # 波形プロット
    plt.subplot(2, 1, 1)
    plt.plot(waveform.cpu().numpy().squeeze())
    plt.title(f"Waveform: {filename}")
    
    # スペクトログラム
    plt.subplot(2, 1, 2)
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(waveform.cpu().numpy().squeeze())), 
        ref=np.max
    )
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram: {filename}")
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()