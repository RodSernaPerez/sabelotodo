from functools import reduce

from sabelotodo.config.news_sources import SOURCES
from sabelotodo.text_finder.extractors.wikipedia_extractor import WikipediaExtractor


class TextFinder:

    def __init__(self, language: str = "es"):
        self.__language = language

    @property
    def wikipedia(self):
        return WikipediaExtractor(self.__language)

    @property
    def language(self):
        return self.__language

    @property
    def extractors(self):
        return [self.wikipedia] + self.newspapers

    @property
    def newspapers(self):
        return [n() for n in SOURCES[self.__language]]

    @language.setter
    def language(self, value):
        # TODO: Raise error if language is not ok
        self.__language = value

    def get_texts(self, topics):
        return reduce(lambda list_, t: list_ + self._get_text_one_topic(t),
                      topics, [])

    def _get_text_one_topic(self, topic):
        return reduce(
            lambda list_, extractor: list_ + extractor(topic),
            self.extractors, [])
