import random
import os
import re
from collections import Counter, defaultdict
from itertools import chain
import pickle

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_logger

def geo_mean(values):
    return np.exp(1.0 / len(values) * sum([np.log(p) for p in values])) 

class DataLoader(object):
    def __init__(self, path, logger, word2id=None):
        self.sentence_list = []
        file_names = os.listdir(path)
        max_len = -1
        min_len = 100
        for file_name in tqdm.tqdm(file_names, path):
            with open(os.path.join(path, file_name), 'r') as F:
                for line in F:
                    sentences = re.findall(r'[\u4e00-\u9fa5，。、；！：（）《》“”？—_]+|\d+', line)
                    sentences = [x.split() for x in re.split(r'[！。？]', ' '.join(sentences))]
                    sentences = [sentence for sentence in sentences
                                 if len(sentence) > 5 and len(sentence) < 200]
                    max_len = max([len(x) for x in sentences] + [max_len])
                    min_len = min([len(x) for x in sentences] + [min_len])
                    self.sentence_list.extend(sentences)
        counter = Counter(chain(*self.sentence_list))
        # logger.warn('*' * 80)
        logger.infov('total sentence: %d; max seq_len: %d; min seq_len: %d; total word: %d;',
                    len(self.sentence_list), max_len, min_len, len(counter))
        self.seq_list, self.word2id, self.id2word = \
            self.build_dataset(self.sentence_list, counter, word2id)
        self.vocab_size = len(self.word2id) + 1
    def build_dataset(self, sentence_list, counter, word2id=None):
        if word2id is None:
            word2id = {word: _id + 1 for _id, word in enumerate(counter.keys()) if counter[word] > 20}
            word2id = defaultdict(int, word2id)
        id2word = {_id: word for word, _id in word2id.items()}
        id2word[0] = 'oov'
        return [[word2id[x] for x in sentence] for sentence in sentence_list], word2id, id2word
    def rtranslate(self, x_seq):
        return [self.id2word[x] for x in x_seq]

class HMM(object):
    def __init__(self, hidden_size, observation_size, batch_size):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.pi = np.array(list(range(hidden_size)), dtype=float)
        # self.pi = np.ones(hidden_size)
        self.pi /= np.sum(self.pi, keepdims=True)
        self.A = np.random.rand(hidden_size, hidden_size)
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        self.B = np.random.rand(hidden_size, observation_size)
        self.B /= np.sum(self.B, axis=1, keepdims=True)
    def optimize(self, corpus, test_corpus, max_iter):
        """
        input:
            corpus: list of list
            max_iter: iteration times
        """
        fig1 = plt.figure(figsize=(18, 6))
        fig2 = plt.figure(figsize=(12, 12))
        fig3 = plt.figure(figsize=(12, 6))
        fig4 = plt.figure(figsize=(12, 6))
        result_perplexity = []
        for step in range(max_iter):
            fig1.gca().cla()
            sns.heatmap(np.log(100000 * self.B + 1), ax=fig1.gca(), cbar=False)
            plt.pause(0.001)
            fig1.savefig('b30.png')
            fig2.gca().cla()
            sns.heatmap(self.A, ax=fig2.gca(), cbar=False)
            plt.pause(0.001)
            fig2.savefig('a30.png')
            fig3.gca().plot(self.pi, label=step)
            fig3.gca().legend(loc='upper left', fontsize='xx-small')
            plt.pause(0.001)
            fig3.savefig('pi30.png')
            
            result_prob = []
            # test
            for seq in test_corpus:
                result_prob.append(self.sequence_perplexity(seq))
            # print(result_prob)
            result_perplexity.append(geo_mean(result_prob))
            fig4.gca().cla()
            fig4.gca().semilogy(result_perplexity, marker='<')   
            for idx, result in enumerate(result_perplexity):
                if idx > 1 and idx != len(result_perplexity) - 1:
                    continue
                fig4.gca().text(idx, result, '%.2f'%result, ha='center', va='bottom', fontsize=10)

            plt.pause(0.001)
            fig4.savefig('perplexity30.png')
            self._optimize(corpus)

    def _optimize(self, corpus):
        corpus_iterator = self._corpus_process(corpus)
        for corpus_unit in corpus_iterator:
            # We conduct a batch-based optimization
            pi_d, pi_n = np.zeros((1,)), np.zeros_like(self.pi)
            a_d, b_d  = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))
            a_n, b_n = np.zeros_like(self.A), np.zeros_like(self.B)

            for x_seq in tqdm.tqdm(corpus_unit, 'optimize'):
                alpha, beta = self._compute_mediate(x_seq)  # (N_x, hidden_size)
                alpha_beta = alpha * beta  # (N_x, hidden_size)
                seq_prob = 1e-50 if np.sum(alpha_beta[0, :]) == 0 else np.sum(alpha_beta[0, :])
                pi_d += np.sum(alpha_beta[0, :]) / seq_prob
                pi_n += alpha_beta[0, :] / seq_prob
                b_d += np.sum(alpha_beta, axis=0)[:, np.newaxis] / seq_prob # sum along the first axis
                for i in range(len(x_seq)):
                    b_n[:, x_seq[i]] += alpha_beta[i, :] / seq_prob
                a_temp = alpha[:-1, :, np.newaxis] * beta[1:, np.newaxis, :]  # i -> j
                a_temp *= np.transpose(self.B[:, x_seq[1:], np.newaxis], axes=(1, 0, 2)) * self.A
                a_temp = np.sum(a_temp, axis=0)
                a_d += np.sum(a_temp, axis=1, keepdims=True) / seq_prob
                a_n += a_temp / seq_prob
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

    def sequence_perplexity(self, x_seq):
        """
        input:
            x_seq: np.array of size (N_x, )
        return:
            prob for x
        """
        # forward pass
        alpha = np.zeros(self.hidden_size, dtype=np.float)
        alpha = self.pi * self.B[:, x_seq[0]]  # (hidden_size, )
        for i in range(1, len(x_seq)):
            # (hidden_size, )
            alpha = np.squeeze(self.B[:, x_seq[i]] * np.matmul(alpha[np.newaxis, :], self.A))
        perplexity = np.exp(- 1.0 / len(x_seq) * np.log(np.sum(alpha)))
        if perplexity == np.inf:
            return 6000
        else:
            return perplexity
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
    # training
    logger = get_logger('hmm')
    datas = DataLoader('../datas/train/', logger)
    # inference_data = DataLoader('../datas/test', logger, word2id=datas.word2id)
    # hmm = HMM(30, datas.vocab_size, 33025)
    # hmm.optimize(datas.seq_list, inference_data.seq_list, 50)

    # pickle.dump({'model': hmm, 'word2id': datas.word2id}, open('model30.pkl', 'wb'))
    
    # inference only
    dt = pickle.load(open('model30.pkl', 'rb'))
    hmm = dt['model']
    word2id = dt['word2id']
    inference_data = DataLoader('../datas/test', logger, word2id=word2id)
    result_prob = dict()
    for seq in inference_data.seq_list:
        result_prob[' '.join(inference_data.rtranslate(seq))] = hmm.sequence_perplexity(seq)
    print(result_prob)
    print(geo_mean(result_prob.values()))
    pickle.dump({
        'result': result_prob
    }, open("result.pkl", 'wb'))