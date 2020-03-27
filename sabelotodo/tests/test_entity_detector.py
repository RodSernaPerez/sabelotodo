from unittest import TestCase
from unittest.mock import patch

import tensorflow as tf
import torch

from sabelotodo.entity_detector import EntitiesDetector


@patch("sabelotodo.entity_detector.AutoTokenizer", spec=True)
@patch("sabelotodo.entity_detector.AutoModelForTokenClassification", spec=True)
class TestEntitiesDetector(TestCase):
    _VALUES_OUTPUT_MODEL_1 = [5, 1, 2, 3]
    _VALUES_OUTPUT_MODEL_2 = [0, 3, 2, 9]
    _VALUES_OUTPUT_MODEL_3 = [0, 1, 6, 3]

    _VALUES_OUTPUT = [_VALUES_OUTPUT_MODEL_1, _VALUES_OUTPUT_MODEL_2, _VALUES_OUTPUT_MODEL_3]
    _TOKENS_INPUT_MODEL = ["A", "B", "C"]

    _LABELS = ["O", "B-1", "I-1", "B-2"]

    def test__predicting(self):
        self.fail()

    def test__decode_tensor_tf(self, mock_model, mock_tokenizer):
        ed = EntitiesDetector("model")
        ed.LABELS = self._LABELS

        self.assertEqual(self._LABELS[0], ed._decode_tensor(tf.constant(self._VALUES_OUTPUT_MODEL_1)))
        self.assertEqual(self._LABELS[3], ed._decode_tensor(tf.constant(self._VALUES_OUTPUT_MODEL_2)))
        self.assertEqual(self._LABELS[2], ed._decode_tensor(tf.constant(self._VALUES_OUTPUT_MODEL_3)))

    def test__decode_tensor_torch(self, mock_model, mock_tokenizer):
        ed = EntitiesDetector("model")
        ed.LABELS = self._LABELS

        self.assertEqual(self._LABELS[0], ed._decode_tensor(torch.tensor(self._VALUES_OUTPUT_MODEL_1)))
        self.assertEqual(self._LABELS[3], ed._decode_tensor(torch.tensor(self._VALUES_OUTPUT_MODEL_2)))
        self.assertEqual(self._LABELS[2], ed._decode_tensor(torch.tensor(self._VALUES_OUTPUT_MODEL_3)))

    def test__get_important_tokens_tf(self, mock_model, mock_tokenizer):
        ed = EntitiesDetector("model")
        ed.LABELS = self._LABELS

        x = [(c[0], tf.constant(c[1])) for c in zip(self._TOKENS_INPUT_MODEL, self._VALUES_OUTPUT)]
        y = ed._get_important_tokens(x)
        self.assertEqual([('B', 'B'), ('C', 'I')], y)

    def test__get_important_tokens_torch(self, mock_model, mock_tokenizer):
        ed = EntitiesDetector("model")
        ed.LABELS = self._LABELS

        x = [(c[0], torch.tensor(c[1])) for c in zip(self._TOKENS_INPUT_MODEL, self._VALUES_OUTPUT)]
        y = ed._get_important_tokens(x)
        self.assertEqual([('B', 'B'), ('C', 'I')], y)

    def test__decode_entities(self, mock_model, mock_tokenizer):

        ed = EntitiesDetector("model")
        ed.LABELS = self._LABELS

        mock_tokenizer.from_pretrained.return_value.decode.side_effect = ["hola", "que", "tal"]
        ids = [1, 2, 3, 4, 5]
        labels = ["B", "I", "B", "B", "I"]

        result = ["hola", "que", "tal"]
        entities = ed._decode_entities(list(zip(ids, labels)))

        mock_tokenizer.from_pretrained.return_value.call_count = 3
        self.assertEqual(result, entities)

    def test_get_entities(self):
        self.fail()
