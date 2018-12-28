import numpy as np


class DataLoader(object):
    def __init__(self, path):
        pass

    def __getitem__(self, item):
        pass


class HMM(object):
    def __init__(self, hidden_size):
        self._pi = np.array(hidden_size, dtype=np.)
        self._a = np.array([hidden_size, hidden_size])
        self._b = np.array([hidden_size,])

    def hidden_state_inference(self, observation_seq):
        pass

    def optimize(self, corpus):
        pass

    def _expectation(self, ):
        pass

    def _maximum(self):
        pass

    def sequence_probability(self):
        pass


if __name__ == '__main__':
