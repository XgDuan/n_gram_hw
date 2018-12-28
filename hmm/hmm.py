import numpy as np


class DataLoader(object):
    def __init__(self, path):
        pass

    def __getitem__(self, item):
        pass


class HMM(object):
    def __init__(self, hidden_size):
        self._pi = np.array(hidden_size)
        self._a = np.array([hidden_size, hidden_size])
        self._b = np.array([hidden_size,])

    def hidden_state_inference(self, observation_seq):
        pass

    def optimize(self, corpus):
        corpus_iterator = self._corpus_process(corpus)
        for corpus_unit in corpus_iterator:
            alpha, beta = self._compute_mediate(corpus_unit)

    def _compute_mediate(self, corpus_mat):
        
        return alpha, beta
    
    def _expectation(self, ):
        pass

    def _maximum(self):
        pass

    def sequence_probability(self):
        pass

    def _corpus_process(self, corpus):
        """
        For real corpus, currently empty
        """
        # For fake datas, the length of all datas are the same
        corpus = np.array(corpus)  # (n_sample, sequence_len)
        return iter([corpus,])


if __name__ == '__main__':
    pass
