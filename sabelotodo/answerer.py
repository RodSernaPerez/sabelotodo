import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class Answerer:
    PAD_TOKEN = "[PAD]"

    def __init__(self, model_name, batch=32):
        self.batch = batch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    @property
    def special_token_ids_to_tokens(self):
        return {x[0]: x[1] for x in zip(self._tokenizer.all_special_tokens, self._tokenizer.all_special_ids)}

    def answer(self, question, text):
        # TODO: Implementar aqui paradoja de la secretaria
        answers_and_scores = [self._answer_on_one_text(question, t) for t in text.split('.')]
        answers_and_scores = [x for x in answers_and_scores if
                              not any([t in x[0] for t in self.special_token_ids_to_tokens.keys()])]
        answers_and_scores = sorted(answers_and_scores, key=lambda x: x[1])
        return answers_and_scores[-1]

    def _answer_on_one_text(self, question, text):
        input_ids = self._tokenizer.encode(question, text)

        token_type_ids = [0 if i <= input_ids.index(self.special_token_ids_to_tokens['[SEP]']) else 1 for i in
                          range(len(input_ids))]

        start_scores, end_scores = self._model(torch.tensor([input_ids]),
                                               token_type_ids=torch.tensor([token_type_ids]))
        answer_ids = input_ids[torch.argmax(start_scores): torch.argmax(end_scores) + 1]
        answer_text = self._tokenizer.decode(answer_ids)
        score = (torch.max(start_scores) + torch.min(end_scores)).data.numpy() / 2
        return answer_text, score

    def _generate_token_type_ids(self, ids):
        return [0 if i <= ids.index(self.special_token_ids_to_tokens['[SEP]']) else 1 for i in range(len(ids))]

    def _pad_input_ids(self, input_ids):
        def pad_sequence(seq, value, length):
            return seq + [value] * (length - len(seq)), [1] * len(seq) + [0] * (length - len(seq))
        max_length = max([len(id_i) for id_i in input_ids])
        ids, masks = zip(*[pad_sequence(x, self.special_token_ids_to_tokens[self.PAD_TOKEN], max_length) for x in input_ids])
        return ids, masks

    def _process_inputs(self, question, list_of_texts):
        """Returns:
            - input_ids: """
        input_ids = [self._tokenizer.encode(question, t) for t in list_of_texts]
        input_ids, attention_mask = self._pad_input_ids(input_ids)

        token_type_ids = [self._generate_token_type_ids(t) for t in input_ids]

        return input_ids, token_type_ids, attention_mask

    def _answer_on_batch(self, question, list_of_texts):
        input_ids = [self._tokenizer.encode(question, t) for t in list_of_texts]
        input_ids, attention_mask = self._pad_input_ids(input_ids)

        token_type_ids = [self._generate_token_type_ids(t) for t in input_ids]

        start_scores, end_scores = self._model(torch.tensor(input_ids),
                                               token_type_ids=torch.tensor(token_type_ids),
                                               attention_mask=torch.tensor(attention_mask))
        start_scores, end_scores = start_scores.data.numpy(), end_scores.data.numpy()
        scores_per_text = [(np.max(start_scores[i]) + np.max(end_scores[i])) / 2 for i in range(len(list_of_texts))]

        best_text_index = np.argmax(scores_per_text)
        answer_ids = input_ids[best_text_index][np.argmax(start_scores[best_text_index]): np.argmax(end_scores[best_text_index]) + 1]
        answer_text = self._tokenizer.decode(answer_ids)
        score = scores_per_text[best_text_index]
        return answer_text, score

    def answer(self, question, text):
        split_texts = text.split('.')
        answers_and_scores = []
        for i in range(0, len(split_texts), self.batch):
            answers_and_scores.append(
                self._answer_on_batch(question, split_texts[i: min(i + self.batch, len(split_texts))]))
        answers_and_scores = [x for x in answers_and_scores if
                              not any([t in x[0] for t in self.special_token_ids_to_tokens.keys()])]
        answers_and_scores = sorted(answers_and_scores, key=lambda x: x[1])
        return answers_and_scores[-1]

    def __call__(self, question, texts):
        return self.answer(question, texts)
