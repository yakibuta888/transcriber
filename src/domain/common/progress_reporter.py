from settings import logger


class ProgressReporter():
    def __init__(self, window, bar_key, status_key, log_key=None):
        self.window = window
        self.bar_key = bar_key
        self.status_key = status_key
        self.log_key = log_key
        self.log_buffer = []
        self.weights = {
            'preprocessing': 20,
            'diarization': 35,
            'transcription': 35,
            'merge': 8,
            'output': 2
        }
        self.completed = {key: 0 for key in self.weights.keys()}
        self.totals = {key: 0 for key in self.weights.keys()}

        # ネストしたupdateオブジェクトを作成
        self.update = self.UpdateManager(self)


    def _update_task(self, task_name, current, status_text):
        """共通の更新処理"""
        self.completed[task_name] = current
        total_progress = 0
        for task, weight in self.weights.items():
            if self.totals[task] > 0:
                task_progress = (self.completed[task] / self.totals[task]) * weight
                total_progress += task_progress
        
        self.window[self.bar_key].update_bar(int(total_progress))
        percent = (total_progress / sum(self.weights.values())) * 100
        display_text = f"{status_text} (全体: {percent:.1f}%)"

        # ステータス表示を更新
        self.window[self.status_key].update(display_text)

        # ログ用multilineにも同じ内容を追加
        if self.log_key:
            self._add_log(display_text)
        
        self.window.refresh()


    class UpdateManager:
        def __init__(self, parent):
            self.parent = parent
            
        @property
        def preprocessing(self):
            return lambda current, detail="": self.parent._update_task('preprocessing', current, f"音声前処理: {detail}")
            
        @property
        def diarization(self):
            return lambda current, detail="": self.parent._update_task('diarization', current, f"話者分離: {detail}")
            
        @property
        def transcription(self):
            return lambda current, detail="": self.parent._update_task('transcription', current, f"文字起こし: {detail}")
            
        @property
        def merge(self):
            return lambda current, detail="": self.parent._update_task('merge', current, f"結果統合: {detail}")
            
        @property
        def output(self):
            return lambda current, detail="": self.parent._update_task('output', current, f"ファイル出力: {detail}")


    def set_totals(self, preprocessing_steps, diar_segments, asr_chunks, merge_segments, output_lines):
        self.totals.update({
            'preprocessing': preprocessing_steps,
            'diarization': diar_segments, 
            'transcription': asr_chunks,
            'merge': merge_segments,
            'output': output_lines
        })
        total_weighted = sum(self.weights.values())
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