from abc import ABC, abstractmethod

class IProgressReporter(ABC):
    def __init__(self):
        self.update = self._make_update_manager()

    @abstractmethod
    def set_totals(self, **step_totals):
        pass

    @abstractmethod
    def set_output_total(self, output_lines):
        pass

    @abstractmethod
    def set_merge_total(self, merge_segments):
        pass

    @abstractmethod
    def set_transcribe_total(self, transcribe_segments):
        pass

    @abstractmethod
    def set_diarization_total(self, diarization_segments):
        pass

    @abstractmethod
    def clear_log(self):
        pass

    @abstractmethod
    def _add_log(self, message):
        pass

    @abstractmethod
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
