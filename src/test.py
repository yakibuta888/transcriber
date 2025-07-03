import torch
from domain.entity.audio_entity import AudioEntity


def test_audio_entity_creation():
    # 正常なケース
    waveform = AudioEntity.new(
        waveform=torch.randn(1, 16000),  # 1秒のモノラル音声
        sample_rate=16000,
        duration=1.0,
        num_channels=1,
        processing_steps=["step1", "step2"],
        metadata={"source": "test.wav"}
    )
    
    assert waveform.sample_rate == 16000
    # assert waveform.duration == 1.0
    # assert waveform.num_channels == 1
    # assert waveform.processing_steps == ["step1", "step2"]
    # assert waveform.metadata["source"] == "test.wav"
    
    # 異常なケース: チャンネル数の不一致
    try:
        AudioEntity.new(
            waveform=torch.randn(2, 16000),  # 2チャンネルの音声
            sample_rate=16000,
            duration=1.0,
            num_channels=1,  # 不一致
            processing_steps=[],
            metadata={}
        )
    except ValueError as e:
        assert str(e) == "Waveform channels do not match num_channels. Expected: 1, Got: 2"
        
    # 1次元データが正常に変換されることの確認
    waveform = torch.randn(1, 16000)
    waveform = waveform.squeeze(0)  # (1, time) → (time,)

    waveform_1d = AudioEntity.new(
        waveform=waveform,
        sample_rate=16000,
        duration=1.0,
        num_channels=1,
        processing_steps=[],
        metadata={}
    )
    
    assert waveform_1d.waveform.dim() == 2  # 1次元データが2次元に変換されていること
    

if __name__ == "__main__":
    test_audio_entity_creation()
    print("All tests passed.")