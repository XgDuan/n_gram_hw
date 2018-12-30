import numpy as np
import random

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
            # We conduct a batch-based optimization
            pi_d, pi_n = np.zeros((1,)), np.zeros_like(self.pi)
            a_d, b_d  = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))
            a_n, b_n = np.zeros_like(self.A), np.zeros_like(self.B)

            for x_seq in corpus_unit:
                alpha, beta = self._compute_mediate(x_seq)  # (N_x, hidden_size)
                alpha_beta = alpha * beta  # (N_x, hidden_size)
                pi_d += np.sum(alpha_beta[0, :])
                pi_n += alpha_beta[0, :]
                b_d += np.sum(alpha_beta, axis=0)[:, np.newaxis]  # sum along the first axis
                for i in range(len(x_seq)):
                    b_n[:, x_seq[i]] += alpha_beta[i, :]
                a_temp = alpha[:-1, :, np.newaxis] * beta[1:, np.newaxis, :]  # N_x - 1, h, h
                a_temp = a_temp * self.A * np.transpose(self.B[:, x_seq[1:], np.newaxis], axes=(1, 0, 2))
                a_temp = np.sum(a_temp, axis=0)
                a_d += np.sum(a_temp, axis=1, keepdims=True)
                a_n += a_temp
            # TODO: add extra check to avoid nan
            # TODO: add logs for the optimization
            self.pi = pi_n / pi_d
            self.A = a_n / a_d
            self.B = b_n / b_d
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
        for i in range(len(x_seq)-2, -1, -1):
            beta[i, :] = np.matmul(self.A, beta[i + 1, :] * self.B[:, x_seq[i + 1]])
        return alpha, beta

    def sequence_probability(self):
        pass

    def _corpus_process(self, corpus):
        """
        input:
            corpus: list of list
        return:
            corpus: list of list of lsit
        """
        # For fake datas, the length of all datas are the same
        random.shuffle(corpus)
        for i in range(len(corpus) // self.batch_size):
            yield corpus[i * self.batch_size: (i+1) * self.batch_size]

if __name__ == '__main__':
    hmm = HMM(10, 16, 2)
    data = [
        [1, 2, 3, 4, 0, 1, 2, 3],
        [1, 3, 5, 9, 1, 0, 2],
        [1, 4, 6, 8 ,2, 0, 3, 7, 11],
        [1, 4, 3, 4 ,2, 0, 3, 7, 12, 14],
        [1, 4, 6, 8 ,2, 0, 3, 7, 11],
    ]
    hmm.optimize(data)
