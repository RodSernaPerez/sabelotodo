import logging
from typing import Dict, List, Tuple

from sabelotodo.answerer import Answerer
from sabelotodo.config.names_models import NAME_MODELS
from sabelotodo.entity_detector import EntitiesDetector
from sabelotodo.text_finder import TextFinder
from sabelotodo.text_piece import TextPiece


class Sabelotodo:
    def __init__(self, language="es"):
        self._language = language

        self._extractor = TextFinder(self._language)
        self._entity_detector = EntitiesDetector(self._models_names["model_ner"])
        self._answerer = Answerer(self._models_names["model_qa"])

    @property
    def _models_names(self) -> Dict:
        try:
            return NAME_MODELS[self._language]
        except KeyError:
            raise ValueError("Language is not supported. Possibilities are: {}".format(NAME_MODELS.keys()))

    def answer(self, question: str) -> Tuple[str, float]:
        entities = self._entity_detector(question)
        if len(entities) < 1:
            raise ValueError("No texts could be found")  # TODO: should not be a value error
        print(entities)
        logging.info("Detected entities: {}".format(entities))
        text_pieces = self._extractor.get_texts(entities)
        for n in text_pieces:
            print(n.title)
        ordered_text_pieces = self._order_text_pieces(text_pieces)
        texts = [t.text for t in ordered_text_pieces]
        return self._answerer(question, texts)

    def _order_text_pieces(self, text_pieces: List[TextPiece]) -> List[TextPiece]:
        # TODO: implement logic to order by importance
        return text_pieces

    def __call__(self, *args, **kwargs):
        return self.answer(*args, **kwargs)
