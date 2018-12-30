import numpy as np


class DataLoader(object):
    def __init__(self, path):
        pass

    def __getitem__(self, item):
        pass


class HMM(object):
    def __init__(self, hidden_size, observation_size, batch_size):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.pi = np.random.rand(hidden_size)  # random initialization
        self.A = np.random.rand(hidden_size, hidden_size)
        self.B = np.random.rand(hidden_size, observation_size)

    def hidden_state_inference(self, observation_seq):
        pass

    def optimize(self, corpus):
        corpus_iterator = self._corpus_process(corpus)
        for corpus_unit in corpus_iterator:
            alpha, beta = self._compute_mediate(corpus_unit)

    def _compute_mediate(self, x_seq):
        """
        input:
            x_seq: np.array of size (N_x, )
        return:
            alpha: np.array of size: (N_x, hidden_size)
            beta: np.array of size: (N_x, hidden_size)
        """
        # forward pass
        alpha = np.zeros((len(x_seq), self.hidden_size), dtype=np.float)
        alpha[0, :] = self.pi * self.B[:, x_seq[0]]  # (hidden_size, )
        for i in range(1, len(x_seq)):
            # (hidden_size, )
            alpha[i, :] = self.B[:, x_seq[i]] * np.matmul(alpha[i-1, np.newaxis, :], self.A)

        # backward pass
        beta = np.zeros_like(alpha)
        beta[-1, :] = 1
        for i in range(len(x_seq)-1, -1, -1):
            beta[i, :] = np.matmul(self.A, beta[i + 1, :, np.newaxis] * self.B[:, x_seq[i + 1]])
        return alpha, beta

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
