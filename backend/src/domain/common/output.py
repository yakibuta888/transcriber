import os

from settings import logger

def write_text_file(outdir: str, outname: str, contents: list[dict[str, str]]):
    try:
        outfile = os.path.join(outdir, f"{outname}.txt")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        contents.sort(key=lambda x: x["start"])  # 開始時間でソート
        with open(outfile, "w", encoding="utf-8") as f:
            for item in contents:
                start = item.get("start", 0)
                end = item.get("end", 0)
                speaker = item.get("speaker", "unknown")
                text = item.get("text", "")
                f.write(f"[{start:.2f} - {end:.2f}] {speaker}: {text}\n")
        logger.info(f"Written transcription to {outfile}")
    except Exception as e:
        logger.error(f"Failed to write transcription file. @output.write_text_file: {e}")
        raise RuntimeError(f"Failed to write transcription file: {e}")