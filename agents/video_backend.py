from abc import ABC, abstractmethod

class VideoBackend(ABC):

    @abstractmethod
    def render(
        self,
        image_path: str,
        prompt: str,
        duration: float
    ):
        pass
