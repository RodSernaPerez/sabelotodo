from unittest import TestCase

from sabelotodo.text_finder.news import ElConfidencial


class TestElConfidencial(TestCase):
    def test_get_texts(self):
        instance = ElConfidencial()
        texts = instance.get_texts("economía")
        self.assertTrue(isinstance(texts, list))
        map(lambda x: self.assertTrue(isinstance(x, str)), texts)
        map(lambda x: self.assertTrue(len(x)), texts)
