import math
from six import iteritems
from six.moves import range

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25

class BM25(object):
    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.
        """
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._initialize(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
                
        for document in corpus:
            self.corpus_size += 1
            cur_doc_len = 0
            frequencies = {}

            for word_tuple in document:
                word, word_cnt = word_tuple[0], word_tuple[1]
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += word_cnt
                cur_doc_len += word_cnt
            self.doc_freqs.append(frequencies)
            self.doc_len.append(cur_doc_len)
            num_doc += cur_doc_len

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in iteritems(nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        eps = EPSILON * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_score(self, document, index):
        """Computes BM25 score of given `document` in relation to item of corpus selected by `index`.
        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.
        Returns
        -------
        float
            BM25 score.
        """
        score = 0
        doc_freqs = self.doc_freqs[index]
        for word_tuple in document:
            word = word_tuple[0]
            if word not in doc_freqs:
                continue
            score += (self.idf[word] * doc_freqs[word] * (PARAM_K1 + 1)
                      / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
        return score

    def get_scores(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.
        Parameters
        ----------
        document : list of str
            Document to be scored.
        Returns
        -------
        list of float
            BM25 scores.
        """
        scores = [self.get_score(document, index) for index in range(self.corpus_size)]
        return scores

    def get_scores_bow(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.
        Parameters
        ----------
        document : list of str
            Document to be scored.
        Returns
        -------
        list of float
            BM25 scores.
        """
        scores = []
        for index in range(self.corpus_size):
            score = self.get_score(document, index)
            if score > 0:
                scores.append((index, score))
        return scores
