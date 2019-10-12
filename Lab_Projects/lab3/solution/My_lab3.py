# author: 'GUPTA, AYUSH' | 'Ayush GUPTA'
import numpy as np
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict


def calculate_bow(corpus):
    """
    Calculate bag of words representations of corpus
    Parameters
    ----------
    corpus: list
        Documents represented as a list of string

    Returns
    ----------
    corpus_bow: list
        List of tuple, each tuple contains raw text and vectorized text
    vocab: list
    """
    # YOUR CODE HERE
    corpus_bow, vocab = [],[]
    vocab = set(word for doc in corpus for word in doc.split())                         
    vocab = list(vocab)

    word2id = dict(zip(vocab, range(len(vocab))))

    #word2id = {}
    #for i , v in enumerate(vocab):
    #    word2id[v] = i

    for doc in corpus:
        bow = np.zeros(len(vocab), dtype = int)
        for word in doc.split():
            bow[word2id[word]] += 1
        corpus_bow.append((doc, bow))
    return corpus_bow, vocab


def calculate_tfidf(corpus, vocab):
    """
    Parameters
    ----------
    corpus: list of tuple
        Output of calculate_bow()
    vocab: list
        List of words, output of calculate_bow()

    Returns
    corpus_tfidf: list
        List of tuple, each tuple contains raw text and vectorized text
    ----------

    """
    

    def termfreq(matrix, doc, term):
        try:
            # YOUR CODE HERE
            bow = matrix[doc][1]
            tf = bow[term] / sum(bow)

            return tf
        except ZeroDivisionError:
            return 0

    def inversedocfreq(matrix, term):
        try:
            # YOUR CODE HERE
            N = len(matrix)
            n_t = 0
            for doc in matrix:
                bow = doc[1]
                if bow[term] > 0:
                    n_t += 1

            idf = N/n_t

            return idf

                    
        except ZeroDivisionError:
            return 0

    # YOUR CODE HERE
    corpus_tfidf = []
    tfidf_mat = np.zeros((len(corpus), len(vocab)), dtype = float)

    for doc_id, doc in enumerate(corpus): # corpus: list of tuple

        for word_id, word_freq in enumerate(doc[1]):
            tf = termfreq(corpus, doc_id, word_id)
            idf = inversedocfreq(corpus, word_id)
            tfidf = tf*idf
            tfidf_mat[doc_id, word_id] = tfidf

    all_sents = list(doc[0] for doc in corpus)
    corpus_tfidf = list (zip(all_sents, tfidf_mat))

    return corpus_tfidf



def cosine_sim(u,v):
    """
    Parameters
    ----------
    u: list of number
    v: list of number

    Returns
    ----------
    cosine_score: float
        cosine similarity between u and v
    """
    # YOUR CODE HERE
    cosine_score = np.dot(u,v) / (sqrt(np.dot(u,u))*sqrt(np.dot(v,v)))
    return cosine_score


def print_similarity(corpus):
    """
    Print pairwise similarities
    """
    for sentx in corpus:
        for senty in corpus:
            print("{:.4f}".format(cosine_sim(sentx[1], senty[1])), end=" ")
        print()
    print()



def q1():
    all_sents = ["this is a foo bar",
                 "foo bar bar black sheep",
                 "this is a sentence"]
    corpus_bow, vocab = calculate_bow(all_sents)
    print(corpus_bow)
    print(vocab)

    print("Test BOW cosine similarity")
    print_similarity(corpus_bow)

    print("Test tfidf cosine similarity")
    corpus_tfidf = list(zip(all_sents, calculate_tfidf(corpus_bow, vocab)))
    corpus_tfidf = calculate_tfidf(corpus_bow, vocab)
    print(corpus_tfidf)
    print_similarity(corpus_tfidf)


if __name__ == "__main__":
    q1()
