import sys
import os

from pathlib import Path


def get_models_path(folder_name: str) -> str:
    if getattr(sys, 'frozen', False):
        # exe化後: 実行ファイルのあるディレクトリ
        base_dir = os.path.dirname(sys.executable)
    else:
        # 開発時: srcを基準にする
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
    model_path = os.path.join(base_dir, 'models', folder_name)
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Models directory does not exist: {model_path}")
    
    model_path_str = model_path.as_posix()
    return model_path_str
