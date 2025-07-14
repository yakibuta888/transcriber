from settings import logger


class ProgressReporter():
    def __init__(self, window, bar_key, status_key, log_key=None, weights=None, step_labels=None):
        self.window = window
        self.bar_key = bar_key
        self.status_key = status_key
        self.log_key = log_key
        self.log_buffer = []
        self.weights = weights or {
            'preprocessing': 20,
            'diarization': 35,
            'transcription': 35,
            'merge': 8,
            'output': 2
        }  # デフォルトの重み付け
        # ステップ表示名のデフォルト
        self.step_labels = step_labels or {
            'preprocessing': '音声前処理',
            'diarization': '話者分離',
            'transcription': '文字起こし',
            'merge': '結果統合',
            'output': 'ファイル出力'
        }
        # 新しいステップにも対応
        for key in self.weights:
            if key not in self.step_labels:
                self.step_labels[key] = key
        self.completed = {key: 0 for key in self.weights.keys()}
        self.totals = {key: 0 for key in self.weights.keys()}
        self.update = self._make_update_manager()

    def _make_update_manager(self):
        # 任意のステップ名で呼び出せるようにする
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
        """共通の更新処理"""
        self.completed[task_name] = current
        total_progress = 0
        # 有効なステップだけで進捗を計算
        active_weights = {k: self.weights[k] for k in self.totals if self.totals[k] > 0}
        for task, weight in active_weights.items():
            if self.totals[task] > 0:
                task_progress = (self.completed[task] / self.totals[task]) * weight
                total_progress += task_progress

        max_weight = sum(active_weights.values())
        self.window[self.bar_key].update_bar(int(total_progress), max=max_weight)
        percent = (total_progress / max_weight) * 100 if max_weight > 0 else 0
        display_text = f"{status_text} (全体: {percent:.1f}%)"

        # ステータス表示を更新
        self.window[self.status_key].update(display_text)

        # ログ用multilineにも同じ内容を追加
        if self.log_key:
            self._add_log(display_text)
        
        self.window.refresh()


    def set_totals(self, **step_totals):
        # step_totals: {'transcription': 100, 'output': 1} のように可変長で渡す
        # 使うステップだけを有効化
        self.totals.update({k: v for k, v in step_totals.items() if k in self.weights})
        self.completed = {k: 0 for k in self.totals}
        total_weighted = sum(self.weights[k] for k in self.totals)
        self.window[self.bar_key].update_bar(0, max=total_weighted)


    def set_output_total(self, output_lines):
        """出力行数を後から設定"""
        self.totals['output'] = output_lines
    
    def set_merge_total(self, merge_segments):
        """マージ処理数を後から設定"""
        self.totals['merge'] = merge_segments
    

    def _add_log(self, message):
        """ログ用multilineに内容を追加（タイムスタンプ付き）"""
        if not self.log_key:
            return
            
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            logger.info(f"Log: {log_message}")

            # 既存のログを取得
            if not self.log_buffer:
                self.log_buffer.append(self.window[self.log_key].get())

            # ログ履歴に追加（最大1000行まで保持）
            self.log_buffer.append(log_message)
            if len(self.log_buffer) > 1000:
                self.log_buffer.pop(0)  # 古いログを削除
            
            # multilineを更新
            log_text = "\n".join(self.log_buffer) + "\n"
            self.window[self.log_key].update(log_text)
            
        except Exception as e:
            print(f"Warning: Could not update log: {e}")

    def clear_log(self):
        """ログをクリア"""
        if self.log_key:
            self.log_buffer.clear()
            self.window[self.log_key].update("")