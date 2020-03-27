import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class EntitiesDetector:
    LABELS = ["B-LOC",
              "B-MISC",
              "B-ORG",
              "B-PER",
              "I-LOC",
              "I-MISC",
              "I-ORG",
              "I-PER",
              "O"]

    @property
    def DICT_ID_TO_LABEL(cls):
        return {i: cls.LABELS[i] for i in range(len(cls.LABELS))}

    def __init__(self, name_model="mrm8488/bert-spanish-cased-finetuned-ner"):
        self.tokenizer_ner = AutoTokenizer.from_pretrained(name_model)
        self.model_ner = AutoModelForTokenClassification.from_pretrained(name_model)

    def _predicting(self, text):
        input_ids = self.tokenizer_ner.encode(text, add_special_tokens=True)

        input_ids_tensor = torch.tensor([input_ids])  # Batch size 1
        labels = torch.tensor([[1] * len(input_ids)])  # Batch size 1
        outputs = self.model_ner(input_ids_tensor, labels=labels,
                                 attention_mask=labels)
        loss, scores = outputs[:2]
        tokens_with_scores = list(zip(input_ids, scores[0]))

        return tokens_with_scores

    def _decode_tensor(self, tensor):
        """Gets the highest value index from the tensor and converts it to the tags."""
        try:
            tensor = tensor.detach() # Needed in torch
        except AttributeError:
            pass
        i = np.argmax(tensor.numpy())
        return self.LABELS[i]

    def _get_important_tokens(self, tokens_with_scores):
        """Gets the tokens that are part of an entity (not classified as an O)."""

        pos_tags = [(k[0], self._decode_tensor(k[1])[0]) for k in tokens_with_scores]
        pos_tags = [p for p in pos_tags if p[1] != "O"]
        return pos_tags

    def _decode_entities(self, pos_tags):
        entities = []
        current_entity = []
        for token, tag in pos_tags:
            if tag == "B" and len(current_entity):
                entities.append(self.tokenizer_ner.decode(current_entity))
                current_entity = []
            current_entity.append(token)
        if len(current_entity):
            entities.append(self.tokenizer_ner.decode(current_entity))

        for i in reversed(range(len(entities))):
            if entities[i].startswith("##"):
                entities[i - 1] = entities[i - 1] + entities[i][2:]
                del entities[i]
        return entities

    def get_entities(self, text):
        tokens_with_scores = self._predicting(text)
        pos_tags = self._get_important_tokens(tokens_with_scores)
        entities = self._decode_entities(pos_tags)

        return entities

    def __call__(self, text):
        return self.get_entities(text)
