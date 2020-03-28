from typing import List

import wikipedia

from sabelotodo.text_finder.extractors.extractor import Extractor
from sabelotodo.text_piece import TextPiece


class WikipediaExtractor(Extractor):
    __SPECIAL_CHAR = "<SEP_TITLES>"

    def __init__(self, language):
        self.language = language

    def _get_one_page(self, topic) -> wikipedia.WikipediaPage:
        try:
            page = wikipedia.page(topic)
        except:
            raise ValueError("No text could be found.")
        return page

    def _get_sections(self, page) -> List[TextPiece]:
        titles = [p for p in page.content.split('\n') if p.startswith("==")]
        long_titles = self._get_long_titles(titles)
        pieces = []
        for t in long_titles:
            title_piece = t.replace("=", "").split(self.__SPECIAL_CHAR)[-1].strip()
            text_piece = page.section(title_piece)
            if len(text_piece):
                pieces.append(
                    TextPiece(title=t.replace("=", "").replace(self.__SPECIAL_CHAR, ", "),
                              text=text_piece))
        return pieces

    def _get_long_titles(self, titles) -> List[str]:
        """Joins the titles with the subtitles in a unique string."""

        def get_level(tit):
            return tit.count("=") / 2

        if not titles:
            return []

        level = get_level(titles[0])

        x = titles[1:]
        next_title_index = len(titles)

        for i, t in enumerate(titles[1:], start=1):
            if get_level(t) == level:
                x = titles[1: i - 1]
                next_title_index = i
                break

        results = [titles[0]] + list(map(lambda t: titles[0] + self.__SPECIAL_CHAR + t, self._get_long_titles(x)))
        results += self._get_long_titles(titles[next_title_index:])
        return results

    def extract_topic(self, topic) -> List[TextPiece]:
        wikipedia.set_lang(self.language)
        try:
            page = self._get_one_page(topic)
            text_pieces = self._get_sections(page)
        except:
            text_pieces = []

        return text_pieces
