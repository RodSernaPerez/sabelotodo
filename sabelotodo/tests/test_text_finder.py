from unittest import TestCase

from sabelotodo.text_finder import TextFinder
from sabelotodo.text_piece import TextPiece


class TestTextFinder(TestCase):
    def test_get_texts(self):
        textfinder = TextFinder("es")

        result = textfinder.get_texts(["Rusia"])

        self.assertTrue(isinstance(result, list))
        self.assertTrue(map(lambda x: isinstance(x, TextPiece), result))
