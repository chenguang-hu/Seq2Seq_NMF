import os
import sys
import sys
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = os.path.join(BASE_DIR, "src")
sys.path.append(src)

from utils.util import ngrams

__all__ = ["distinct_n_sentence_level", "distinct_n_corpus_level"]

def cal_Distinct(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity