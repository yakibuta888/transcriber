import os
import argparse

from src.application.factories.create_transcriber import create_transcriber
from src.application.services.transcribe_service import TranscribeService
from src.domain.common.progress_reporter_cui import ProgressReporterCUI
from src.settings import CustomLoguru, WarningLogger, logger


def main():
    parser = argparse.ArgumentParser(description="文字起こしアプリ（CUI版）")
    parser.add_argument('--model', required=True, choices=['whisper-large-v3', 'kotoba-whisper-v2.2'], help='モデル名')
    parser.add_argument('--infile', required=True, help='入力音声ファイルパス')
    parser.add_argument('--outdir', default=None, help='出力フォルダ（省略時は入力ファイルと同じ）')
    parser.add_argument('--outname', default='', help='出力ファイル名')
    # 詳細オプション
    parser.add_argument('--diarize', action='store_true', help='話者分離を有効にする')
    parser.add_argument('--add_punctuation', action='store_true', help='句読点付与')
    parser.add_argument('--num_speakers', type=int, default=None, help='話者数')
    parser.add_argument('--min_speakers', type=int, default=None, help='話者数の最小値')
    parser.add_argument('--max_speakers', type=int, default=None, help='話者数の最大値')
    parser.add_argument('--add_silence_start', type=float, default=None, help='先頭無音（秒）')
    parser.add_argument('--add_silence_end', type=float, default=None, help='末尾無音（秒）')
    parser.add_argument('--chunk_length', type=int, default=15, help='チャンク長（秒）')
    parser.add_argument('--batch_size', type=int, default=8, help='バッチサイズ')
    parser.add_argument('--fa2', action='store_true', help='Flash Attention 2利用')
    parser.add_argument('--hf_token', default='', help='HuggingFaceトークン')
    args = parser.parse_args()

    infile = args.infile
    outdir = args.outdir if args.outdir else os.path.dirname(infile)
    outname = args.outname
    model = args.model
    hf_token = args.hf_token

    option_args = {
        'diarize': args.diarize,
        'add_punctuation': args.add_punctuation,
        'num_speakers': args.num_speakers,
        'min_speakers': args.min_speakers,
        'max_speakers': args.max_speakers,
        'add_silence_start': args.add_silence_start,
        'add_silence_end': args.add_silence_end,
        'chunk_length': args.chunk_length,
        'batch_size': args.batch_size,
        'fa2': args.fa2,
    }

    # 進捗レポーター
    progress_reporter = ProgressReporterCUI()

    print("=== 文字起こし処理開始 ===")
    try:
        transcriber = create_transcriber(
            audio_file=infile,
            model=model,
            hf_token=hf_token
        )
        transcribe_service = TranscribeService(transcriber=transcriber)
        transcribe_service.transcribe_and_save(
            outdir=outdir,
            outname=outname,
            option_args=option_args,
            progress=progress_reporter
        )
        print("\n=== 処理完了 ===")
    except ValueError as e:
        logger.error(f"Transcription failed. @main: {e}", exc_info=True)
        print("エラー: モデル選択が不正です。管理者にお問い合わせください。")
    except RuntimeError as e:
        logger.error(f"Transcription failed. @main: {e}", exc_info=True)
        print("エラー: 音声認識に失敗しました。入力ファイルを確認してください。")
    except FileNotFoundError as e:
        logger.error(f"File not found. @main: {e}", exc_info=True)
        print("エラー: 入力ファイルが見つかりません。ファイルパスを確認してください。")
    except Exception as e:
        logger.error(f"Unexpected error occurred. @main: {e}", exc_info=True)
        print(f"予期しないエラーが発生しました。管理者にお問い合わせください。")

if __name__ == "__main__":
    main()
