from unittest import TestCase
from unittest.mock import patch

from sabelotodo.answerer import Answerer


class TestAnswerer(TestCase):

    @patch("sabelotodo.answerer.AutoModelForQuestionAnswering")
    @patch("sabelotodo.answerer.AutoTokenizer")
    def test__pad_input_ids(self, mock_tokenizer, mock_model):
        x_1 = [1, 2, 3, 4]
        x_2 = [1, 2]

        y_1 = [1, 2, 3, 4]
        y_2 = [1, 2, 0, 0]
        m_1 = [1, 1, 1, 1]
        m_2 = [1, 1, 0, 0]

        mock_tokenizer.from_pretrained.return_value.all_special_tokens = ["[PAD]"]
        mock_tokenizer.from_pretrained.return_value.all_special_ids = [0]
        ans = Answerer("m")
        self.assertEqual(ans._pad_input_ids([x_1, x_2]),
                         ((y_1, y_2), (m_1, m_2)))
