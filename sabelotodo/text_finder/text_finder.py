import wikipedia

wikipedia.set_lang("es")


class TextFinder:

    @classmethod
    def _get_one_page(cls, topic):
        try:
            text = wikipedia.page(topic).content
        except:
            raise ValueError("No text cold be found.")
        return text

    @classmethod
    def get_texts(cls, topics):
        text = "\n".join([cls._get_one_page(t) for t in topics])
        return text
