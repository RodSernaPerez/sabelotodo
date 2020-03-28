from unittest import TestCase

from sabelotodo.text_finder.extractors.wikipedia_extractor import WikipediaExtractor
from sabelotodo.text_piece import TextPiece


class TestWikipediaExtractor(TestCase):
    def test_get_texts_spanish(self):
        we = WikipediaExtractor("es")
        x = we(["Rusia"])
        self.assertTrue(isinstance(x, list))
        map(lambda n: self.assertTrue(isinstance(n, TextPiece)), x)
