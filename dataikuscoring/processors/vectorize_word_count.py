from ..utils import Tokenizer
from .preprocessor import Preprocessor


class VectorizeWordCount(Preprocessor):

    FILENAME = "word_counts"

    def __init__(self, parameters):
        self.tokenizers = [
            Tokenizer(stop_words, min_n_grams, max_n_grams)
            for stop_words, min_n_grams, max_n_grams in zip(
                parameters["stop_words"],
                parameters["min_n_grams"],
                parameters["max_n_grams"],
            )
        ]
        self.columns = parameters["column"]
        self.vocabulary = [set(x) for x in parameters["vocabulary"]]
        self.unrecorded_value = parameters["unrecorded_value"]

    def process(self, X_numeric, X_non_numeric):
        for column, tokenizer, vocab in zip(self.columns, self.tokenizers, self.vocabulary):
            token_counts = [tokenizer.get_token_counts(text if text is not None else "") for text in X_non_numeric[:, column]]
            # input matrix initialization
            tokens = set()
            for token_count in token_counts:
                tokens.update(token_count.keys())
            for token in tokens:
                X_numeric[:, "countvec:{}:{}".format(column, token)] = self.unrecorded_value
            for (index, token_count) in enumerate(token_counts):
                for token, count in token_count.items():
                    if token in vocab:
                        X_numeric[index, "countvec:{}:{}".format(column, token)] = count
        return X_numeric, X_non_numeric

    def __repr__(self):
        return "CountVectorizer({})".format(", ".join(self.columns))
