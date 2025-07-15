import logging.config
import os
import yaml
from pathlib import Path


CWD: Path = Path(__file__).resolve().parent
LOG_CONFIG_PATH: str = os.path.normpath(os.path.join(CWD, "log/log_config.yaml"))

def setup_logging() -> None:
    with open(LOG_CONFIG_PATH, 'r', encoding='utf-8') as f:
        log_config = yaml.safe_load(f)

        # GitHub Actionsや他のCI環境での実行を検出
        if os.environ.get('CI'):  # CI環境であればTrue
            # ファイルハンドラを削除し、コンソールハンドラのみを使用
            log_config['handlers'].pop('file', None)
            for logger in log_config['loggers'].values():
                logger['handlers'] = [handler for handler in logger['handlers'] if handler != 'file']

        logging.config.dictConfig(log_config)  # type: ignore


setup_logging()
logger = logging.getLogger(__name__)