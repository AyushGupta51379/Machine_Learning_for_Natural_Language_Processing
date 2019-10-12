import nltk
import numpy as np
import timeit
import time
import itertools

from scipy import sparse
# switch between gutenberg, webtext, brown, reuters
from nltk.corpus import reuters as data_holder
from sklearn.feature_extraction.text import TfidfVectorizer


class VecSpaceModel:
    def __init__(self, data_holder):
        self.data = self.load_data(data_holder)
        self.word2id = self.get_vocabulary(self.data)
        self.bow, self.bow_norm = self.calculate_bow()
        self.tfidf, self.tfidf_norm = self.calculate_tfidf()
        print("TfIdf mat shape for {:} : {:}".format(self.__class__.__name__, self.tfidf.shape))

    def load_data(self, data_holder):
        docs = [list(data_holder.words(doc_fileid)) for doc_fileid in data_holder.fileids()]
        return docs

    def calculate_bow(self,):
        pass

    def calculate_tfidf(self,):
        pass

    def calculate_cos(self, x, y, mode='bow'):
        pass

    def calculate_all_cos(self, mode='bow'):
        pass

    def get_vocabulary(self, docs):
        """
        Parameters
        ----------
        docs: list
            a list of words

        Returns
        ----------
        vocab: list
            unique words in docs
        """
        vocab = sorted(set(list(itertools.chain(*docs))))

        word2id = dict(zip(vocab,range( len(vocab))))
        return word2id


class SparseVec(VecSpaceModel):
    # Sparse matrix: https://docs.scipy.org/doc/scipy/reference/sparse.html
    def __init__(self, data):
        super().__init__(data)

    def calculate_bow(self):
        # lil matrix https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
        row_len, col_len = len(self.data), len(self.word2id)
        data_xit = sparse.lil_matrix((row_len, col_len))
        for i, doc in enumerate(self.data):
            for word in doc:
                word_idx = self.word2id.get(word, -1)
                if word_idx != -1:
                    data_xit[i, word_idx] += 1
        data_xit = data_xit.tocsr()
        norm = sparse.linalg.norm(data_xit, 2, axis=1)

        return data_xit, norm

    def calculate_tfidf(self, ):
        data_csr = self.bow
        row_len, col_len = data_csr.get_shape()
        df = data_csr.getnnz(1)
        idf = np.expand_dims(np.log2(col_len/df), 1)

        tfidf = data_csr.multiply(idf)
        norm = sparse.linalg.norm(tfidf, 2, axis=1)

        return tfidf, norm


    def calculate_cos(self, x, y, mode='bow'):
        if mode == 'bow':
            mat, norm = self.bow, self.bow_norm
        elif mode == 'tfidf':
            mat, norm = self.tfidf, self.tfidf_norm
        else:
            raise NotImplementedError()
        vx = mat[x]
        vy = mat[y].transpose()
        cos = vx.dot(vy)/norm[x]/norm[y]
        return cos

    def calculate_all_cos(self, mode='bow'):

        if mode == 'bow':
            mat, norm = self.bow, self.bow_norm
        elif mode == 'tfidf':
            mat, norm = self.tfidf, self.tfidf_norm
        else:
            raise NotImplementedError()

        dot = mat.dot(mat.transpose())
        cos = dot/np.outer(norm, norm)
        return cos


class DensVec(VecSpaceModel):
    def __init__(self, data):
        super().__init__(data)

    def calculate_bow(self,):
        row_len, col_len = len(self.data), len(self.word2id)
        data_xit = np.zeros((row_len, col_len))
        for i, doc in enumerate(self.data):
            for word in doc:
                word_idx = self.word2id.get(word, -1)
                if word_idx != -1:
                    data_xit[i, word_idx] += 1
        norm = np.linalg.norm(data_xit, 2, axis=1)
        return data_xit, norm

    def calculate_tfidf(self,):
        row_len, col_len = self.bow.shape
        df = np.count_nonzero(self.bow, 1)
        idf = np.expand_dims(np.log2(col_len/df), 1)

        tfidf = np.multiply(self.bow,idf)
        norm = np.linalg.norm(tfidf, 2, axis=1)

        return tfidf, norm

    def calculate_cos(self, x, y, mode='bow'):
        if mode == 'bow':
            mat, norm = self.bow, self.bow_norm
        elif mode == 'tfidf':
            mat, norm = self.tfidf, self.tfidf_norm
        else:
            raise NotImplementedError()
        vx = mat[x]
        vy = mat[y]

        return np.dot(vx, vy)/norm[x]/norm[y]

    def calculate_all_cos(self, mode='bow'):
        if mode == 'bow':
            mat, norm = self.bow, self.bow_norm
        elif mode == 'tfidf':
            mat, norm = self.tfidf, self.tfidf_norm
        else:
            raise NotImplementedError()

        dot = np.dot(mat, mat.transpose())
        cos = dot/np.outer(norm, norm)
        return cos

class SKLearnVec(SparseVec):
    def __init__(self, data_holder):

        vectorizer = TfidfVectorizer(lowercase=False, strip_accents=None, stop_words=None, norm='l2', smooth_idf=False, )
        docs = [data_holder.raw(doc_fileid) for doc_fileid in data_holder.fileids()]
        self.tfidf = vectorizer.fit_transform(docs)
        self.tfidf_norm =  sparse.linalg.norm(self.tfidf, 2, axis=1)
        print("TfIdf mat shape for {:} : {:}".format(self.__class__.__name__, self.tfidf.shape))


    def calculate_cos(self, x, y, mode='bow'):
        super().calculate_all_cos(x,y,mode)

    def calculate_all_cos(self, mode='bow'):
        if mode == 'bow':
            raise NotImplementedError()
        super().calculate_all_cos(mode)



if __name__=='__main__':
    start = time.time()
    sparse_model = SparseVec(data_holder)
    print('Load sparse:', time.time()-start)

    start = time.time()
    dens_model = DensVec(data_holder)
    print('Load dense:', time.time()-start)

    start = time.time()
    sklearn_model = SKLearnVec(data_holder)
    print('Load sklearn:', time.time()-start)

    run_count = 1
    print('Profiling for {:} runs'.format(run_count))

    print('Sparse matrix:', timeit.timeit('sparse_model.calculate_all_cos("tfidf")', number=run_count, setup='from __main__ import sparse_model'))
    print('Dense matrix:', timeit.timeit('dens_model.calculate_all_cos("tfidf")', number=run_count, setup='from __main__ import dens_model'))
    print('SKLearn matrix:', timeit.timeit('sklearn_model.calculate_all_cos("tfidf")', number=run_count, setup='from __main__ import sklearn_model'))
