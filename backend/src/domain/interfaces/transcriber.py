from abc import ABC, abstractmethod

from domain.common.progress_reporter import ProgressReporter


class ITranscriber(ABC):
    @abstractmethod
    def run(self, option_args: dict, progress: ProgressReporter | None = None) -> list[dict]:
        pass