import json
import os
import sys
import torch
from transformers.pipelines import pipeline

from domain.common.get_models_dir import get_models_dir


def transcribe(local: bool, infile: str, outdir: str, outname: str = "transcription", add_punctuation: bool = True, num_speakers: int | None = None, min_speakers: int | None = None, max_speakers: int | None = None, add_silence_start: float | None = None, add_silence_end: float | None = None, chunk_length_s: int = 15, batch_size: int = 8, flash_attention: bool = False):
    """Transcribe audio file using kotoba-whisper-v2.2 model.
    Args:
        infile (str): Path to the input audio file.
        outdir (str): Path to the output directory.
        outname (str): Name of the output JSON file. Defaults to "transcription".
        add_punctuation (bool): Whether to add punctuation to the transcription.
        num_speakers (int, optional): Fixed number of speakers. Defaults to None.
        min_speakers (int, optional): Minimum number of speakers. Defaults to None.
        max_speakers (int, optional): Maximum number of speakers. Defaults to None.
        add_silence_start (float, optional): Whether to add silence at the start of the audio. Defaults to None.
        add_silence_end (float, optional): Whether to add silence at the end of the audio. Defaults to None.
        chunk_length_s (int): Length of audio chunks in seconds. Defaults to 15.
        batch_size (int): Batch size for processing. Defaults to 8.
        flash_attention (bool): Whether to use flash attention. Defaults to False.
    Returns:
        dict: Transcription result.
    """

    outfile = os.path.join(outdir, f"{outname}.json")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # attn_implementationの設定
    if flash_attention:
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"

    if local:
        selected_model = os.path.join(get_models_dir(), "kotoba-whisper-v2.2")
    else:
        selected_model = "kotoba-tech/kotoba-whisper-v2.2"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": attn_impl} if torch.cuda.is_available() else {}

    pipe = pipeline(
        model=selected_model,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
        trust_remote_code=True,
    )

    # pipeの引数として渡す辞書を作成
    pipe_kwargs = {
        "chunk_length_s": chunk_length_s,
        "add_punctuation": add_punctuation,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "add_silence_start": add_silence_start,
        "add_silence_end": add_silence_end,
    }

    # None,空文字の値は渡さない
    pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v not in [None, ""]}

    result = pipe(infile, **pipe_kwargs)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("＊.｡.＊ﾟ＊.｡.＊ﾟ＊.｡.＊ﾟ＊.｡.＊ﾟ＊.｡.＊ﾟ＊.｡.＊ﾟ＊.｡.＊ﾟ＊.｡.＊ﾟ")
    print(f"Transcription completed. Results saved to {outfile}")

