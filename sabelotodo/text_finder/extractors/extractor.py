import abc
from abc import ABC
from typing import List

from sabelotodo.text_piece import TextPiece


class Extractor(ABC):
    @abc.abstractmethod
    def extract_topic(self, topic) -> List[TextPiece]:
        pass

    def __call__(self, topic) -> List[TextPiece]:
        return self.extract_topic(topic)
