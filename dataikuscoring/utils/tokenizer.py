import re
from collections import Counter


word_token_expression = "(?u)\\b\\w\\w+\\b"


class Tokenizer:

    def __init__(self, stop_words, min_n_grams, max_n_grams, token_expression=word_token_expression, to_lower_case=True):
        self.token_expression = token_expression
        self.stop_words = set([x.lower() if to_lower_case else x for x in stop_words])
        self.to_lower_case = to_lower_case
        self.min_n_grams = min_n_grams
        self.max_n_grams = max_n_grams

    # The list of tokens will be sorted by tokens of same number of token length.
    # This is not a problem given that this function is only used by the count_tokens
    # and thus the sorting of the list does not matter
    def get_token_list(self, document):

        if self.to_lower_case:
            document = document.lower()

        tokens = [token for token in re.findall(word_token_expression, document)
                  if token not in self.stop_words]

        ngrams_list = []
        # generate ngrams for n between min_n_grams and max_n_grams
        for n in range(self.min_n_grams, self.max_n_grams + 1):
            ngrams = zip(*[tokens[i:] for i in range(n)])
            ngrams_list = ngrams_list + [" ".join(ngram) for ngram in ngrams]

        return ngrams_list

    def get_token_counts(self, document):
        token_list = self.get_token_list(document)
        return dict(Counter(token_list))
