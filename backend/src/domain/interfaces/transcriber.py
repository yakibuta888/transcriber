from abc import ABC, abstractmethod

from src.domain.interfaces.progress_reporter import IProgressReporter


class ITranscriber(ABC):
    @abstractmethod
    def run(self, option_args: dict, progress: IProgressReporter | None = None) -> list[dict]:
        pass