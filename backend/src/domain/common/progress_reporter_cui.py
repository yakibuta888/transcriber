import sys

from src.domain.interfaces.progress_reporter import IProgressReporter
from src.settings import logger


class ProgressReporterCUI(IProgressReporter):
    def __init__(self, weights=None, step_labels=None):
        self.log_buffer = []
        self.weights = weights or {
            'preprocessing': 20,
            'diarization': 35,
            'transcription': 35,
            'merge': 8,
            'output': 2
        }
        self.step_labels = step_labels or {
            'preprocessing': '音声前処理',
            'diarization': '話者分離',
            'transcription': '文字起こし',
            'merge': '結果統合',
            'output': 'ファイル出力'
        }
        for key in self.weights:
            if key not in self.step_labels:
                self.step_labels[key] = key
        self.completed = {key: 0 for key in self.weights.keys()}
        self.totals = {key: 0 for key in self.weights.keys()}
        self.update = self._make_update_manager()

    def _make_update_manager(self):
        class UpdateManager:
            def __init__(self, parent):
                self.parent = parent
            def __getattr__(self, step):
                if step not in self.parent.weights:
                    raise AttributeError(f"Unknown step: {step}")
                label = self.parent.step_labels.get(step, step)
                return lambda current, detail="": self.parent._update_task(
                    step, current, f"{label}: {detail}"
                )
        return UpdateManager(self)

    def _update_task(self, task_name, current, status_text):
        self.completed[task_name] = current
        total_progress = 0
        active_weights = {k: self.weights[k] for k in self.totals if self.totals[k] > 0}
        for task, weight in active_weights.items():
            if self.totals[task] > 0:
                task_progress = (self.completed[task] / self.totals[task]) * weight
                total_progress += task_progress
        max_weight = sum(active_weights.values())
        percent = (total_progress / max_weight) * 100 if max_weight > 0 else 0

        display_text = f"{status_text} (全体: {percent:.1f}%)"
        self._print_progress_bar(percent, status_text)
        self._add_log(display_text)

    def _print_progress_bar(self, percent, status_text):
        bar_length = 50
        filled_length = int(round(bar_length * percent / 100))
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r[{bar}] {percent:.1f}% | {status_text}", end='', flush=True)
        if percent >= 100:
            print('')  # 改行

    def set_totals(self, **step_totals):
        self.totals.update({k: v for k, v in step_totals.items() if k in self.weights})
        self.completed = {k: 0 for k in self.totals}
        # 初期状態のバー表示
        self._print_progress_bar(0, "進捗開始")

    def set_output_total(self, output_lines):
        self.totals['output'] = output_lines

    def set_merge_total(self, merge_segments):
        self.totals['merge'] = merge_segments

    def _add_log(self, message):
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            logger.info(f"Log: {log_message}")
            self.log_buffer.append(log_message)
            if len(self.log_buffer) > 1000:
                self.log_buffer.pop(0)
            # ログは適宜print（or必要ならファイル保存等）
            # print(f"\n{log_message}")
        except Exception as e:
            print(f"Warning: Could not update log: {e}", file=sys.stderr)

    def clear_log(self):
        self.log_buffer.clear()

