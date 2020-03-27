import logging
from sabelotodo.answerer import Answerer
from sabelotodo.entity_detector import EntitiesDetector
from sabelotodo.config.names_models import NAME_MODELS
from sabelotodo.text_finder import TextFinder


class Sabelotodo:
    def __init__(self, language="es"):
        self._language = language

        self._entity_detector = EntitiesDetector(self._models_names["model_ner"])
        self._answerer = Answerer(self._models_names["model_qa"])

    @property
    def _models_names(self):
        try:
            return NAME_MODELS[self._language]
        except KeyError:
            raise ValueError("Language is not supported. Possibilities are: {}".format(NAME_MODELS.keys()))

    def answer(self, question):
        entities = self._entity_detector(question)
        print("Entities: {}".format(entities))
        if len(entities) < 1:
            return "No texts could be found"
        logging.info("Detected entities: {}".format(entities))
        texts = TextFinder.get_texts(entities)
        print(texts)
        return self._answerer(question, texts)

    def __call__(self, *args, **kwargs):
        return self.answer(*args, **kwargs)