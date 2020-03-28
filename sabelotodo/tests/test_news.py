from unittest import TestCase

from sabelotodo.text_finder.extractors.news import ElConfidencial
from sabelotodo.text_piece import TextPiece


class TestElConfidencial(TestCase):
    def test_extract_topic(self):
        instance = ElConfidencial()
        texts = instance.extract_topic("economía")
        self.assertTrue(isinstance(texts, list))
        map(lambda x: self.assertTrue(isinstance(x, TextPiece)), texts)
