import importlib.util
import logging.config
import os
import re
import sys
import warnings
import yaml
from loguru import logger as loguru_logger
from pathlib import Path


CWD: Path = Path(__file__).resolve().parent
LOG_CONFIG_PATH: str = os.path.normpath(os.path.join(CWD, "log/log_config.yaml"))


class CustomLoguru:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self._patch_deepfilternet_logger()

    def _logging_handler(self, message):
        record = message.record
        self.logger.log(record['level'].no, record['message'])

    def _patch_deepfilternet_logger(self):
        try:
            # DeepFilterNetのloggerをimport
            logger_path = Path("src/models/DeepFilterNet-0.5.6/DeepFilterNet/df/logger.py")
            spec = importlib.util.spec_from_file_location("df.logger", logger_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec or loader for {logger_path}")
            df_logger = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(df_logger)
            # DeepFilterNetのlogger初期化を上書き
            def custom_init_logger(file=None, level="INFO", model=None):
                df_logger.logger.remove()
                # ファイルへの出力を指定したい場合（必要ならfile追加）
                if file:
                    df_logger.logger.add(file, level=level)
                # 標準loggingにパイプ（これが主用途）
                df_logger.logger.add(self._logging_handler, level=level)
                # 必要に応じて、modelやその他情報のログ記録も
                if model is not None:
                    df_logger.logger.info("Loading model settings of {}".format(model))
            # 上書き
            setattr(df_logger, "init_logger", custom_init_logger)
            # Optionally、loggerのremove/addをアプリ全体にも再適用
            loguru_logger.remove()
            loguru_logger.add(self._logging_handler)
        except ImportError:
            # DeepFilterNetが未インポートの場合は何もしない
            pass


class WarningLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.apply()

    def _custom_showwarning(self, message, category, filename, lineno, file=None, line=None):
        if "vendor" in filename:
            return
        self.logger.warning(
            f"{category.__name__}: {message} (in {filename}:{lineno})"
        )

    def apply(self):
        # showwarningを書き換え
        warnings.showwarning = self._custom_showwarning
        # filterwarnings: warningsを有効化。必要に応じて制御
        # warnings.filterwarnings("default")
        # 特定のWarningだけに絞りたい場合はここを調整
        warnings.filterwarnings("ignore", category=DeprecationWarning)


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

base_name: str = os.path.basename(sys.argv[0])
is_cui: bool = re.search(r'main_cui\.py$', base_name) is not None

if is_cui:
    logger = logging.getLogger("cui")
else:
    logger = logging.getLogger("app")

WarningLogger(logger)
CustomLoguru(logger)
