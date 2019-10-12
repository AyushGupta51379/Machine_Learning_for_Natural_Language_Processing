# author: ‘Your name"
# student_id: "Your student ID"
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
        except ZeroDivisionError:
            return 0

    def inversedocfreq(matrix, term):
        try:
            # YOUR CODE HERE
        except ZeroDivisionError:
            return 0

    # YOUR CODE HERE

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
    # corpus_tfidf = list(zip(all_sents, calculate_tfidf(corpus_bow, vocab)))
    corpus_tfidf = calculate_tfidf(corpus_bow, vocab)
    print(corpus_tfidf)
    print_similarity(corpus_tfidf)


if __name__ == "__main__":
    q1()
