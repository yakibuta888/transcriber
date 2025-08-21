import logging
import time

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout

# 表示するログ情報を保持するクラス
class LogDisplay:
    def __init__(self):
        self.fixed_info = ""      # infoログの最新1件
        self.debug_logs = []      # debugログ全件（流れる）

    def add_log(self, level, message):
        if level == logging.INFO:
            self.fixed_info = message
        elif level == logging.DEBUG:
            self.debug_logs.append(message)

    def make_layout(self):
        layout = Layout()
        # infoログ（固定部）
        info_panel = Panel(self.fixed_info, title="Info", style="bold white on blue")
        layout.split_column(
            Layout(info_panel, name="header", size=3),
            Layout(name="body")
        )
        # debugログ（流動部）
        table = Table()
        table.add_column("Debug Log")
        for log in self.debug_logs[-10:]:  # 直近10件だけ表示（任意で調整）
            table.add_row(log)
        debug_panel = Panel(table, title="Debug", border_style="green")
        layout["body"].update(debug_panel)
        return layout

# 独自ハンドラーを使ってログをLogDisplayに渡す
class RichDisplayHandler(logging.Handler):
    def __init__(self, display):
        super().__init__()
        self.display = display

    def emit(self, record):
        msg = self.format(record)
        self.display.add_log(record.levelno, msg)

display = LogDisplay()
