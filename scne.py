import re
import numpy as np

class SCNE():
    def __init__(self, path):
        ngram2index, vectors = self._read_char_ngram_embeddings(path)
        num, edim = vectors.shape
        self.ngram2index = ngram2index
        self.vectors = vectors
        self.num = num
        self.edim = edim

    def _read_char_ngram_embeddings(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        num, edim = lines[0].strip().split()
        num = int(re.sub("[^0-9]","",num))
        edim = int(re.sub("[^0-9]","",edim))
        vectors = np.empty((num,edim))
        ngram2index = dict()
        for i in range(1, 1+num):
            ngram, vec = lines[i].strip().split(' ', 1)
            ngram2index[ngram] = i-1
            vec = np.fromstring(vec, dtype=float, sep=' ')
            vectors[i-1] = vec
        print("Loaded vectors.shape : ", vectors.shape)
        return ngram2index, vectors

    def _get_substrings(self, input_string):
        length = len(input_string)
        return [input_string[i:j+1] for i in range(length) for j in range(i,length)]

    def get_vector(self, input_string):
        vec = np.zeros(self.edim)
        for subword in self._get_substrings(input_string):
            if subword in self.ngram2index:
                vec += self.vectors[self.ngram2index[subword]]
        return vec
