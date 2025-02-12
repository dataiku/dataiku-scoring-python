import numpy as np

from ..utils import Tokenizer
from .preprocessor import Preprocessor


class VectorizeTfidf(Preprocessor):

    FILENAME = "tfidf"

    def __init__(self, parameters):
        self.tokenizers = [Tokenizer(stop_words, min_n_grams, max_n_grams)
                           for stop_words, min_n_grams, max_n_grams
                           in zip(parameters["stop_words"], parameters["min_n_grams"], parameters["max_n_grams"])]
        self.columns = parameters["column"]
        self.idf = parameters["idf"]
        self.normalization = parameters["norm"][0]

        self.vocabularies = []
        self.output_names = []

        raw_vocabulary = parameters["vocabulary"]

        for i in range(len(raw_vocabulary)):
            voc = {}
            out = {}

            for j in range(len(raw_vocabulary[i])):
                voc[raw_vocabulary[i][j]] = parameters["idf"][i][j]
                out[raw_vocabulary[i][j]] = parameters["output_names"][i][j]

            self.vocabularies.append(voc)
            self.output_names.append(out)
        self.unrecorded_value = parameters["unrecorded_value"]

    def process(self, X_numeric, X_non_numeric):
        for column, tokenizer, vocab, output_name in zip(self.columns, self.tokenizers, self.vocabularies, self.output_names):
            token_counts = [tokenizer.get_token_counts(text if text is not None else "") for text in X_non_numeric[:, column]]
            # input matrix initialization
            for output in output_name.values():
                X_numeric[:, output] = self.unrecorded_value
            for (index, token_count) in enumerate(token_counts):
                norm = 0.0
                for token, count in token_count.items():
                    if token not in vocab:
                        continue
                    idf = vocab[token]
                    value = count * idf

                    if self.normalization == u'L1':
                        norm += abs(value)

                    if self.normalization == u'L2':
                        norm += value ** 2

                if norm <= 0.0:
                    norm = 1.0

                if self.normalization == u'L2':
                    norm = norm ** 0.5

                norm = 1.0 / norm

                for token, count in token_count.items():
                    if token not in vocab:
                        continue

                    idf = vocab[token]
                    value = norm * count * idf
                    X_numeric[index, output_name[token]] = value

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "TfidfVectorizer({})".format(", ".join(self.columns))
