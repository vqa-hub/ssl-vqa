"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import argparse
import sys
import json
import _pickle as cPickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
from utils import get_sent_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vqacp2', help='vqacp2 or vqacp1')
    parser.add_argument('--dataroot', type=str, default='vqacp2/annotations/')
    args = parser.parse_args()
    return args


def create_dictionary(dataroot, task='vqacp2'):
    dictionary = Dictionary()
    if task == 'vqacp2':
        files = [
            'vqacp_v2_test_questions.json',
            'vqacp_v2_train_questions.json'
        ]
        for path in files:
            question_path = os.path.join(dataroot, path)
            qs = json.load(open(question_path))
            for q in qs:
                dictionary.tokenize(q['question'], True)
    else:
        files = [
            'vqacp_v1_test_questions.json',
            'vqacp_v1_train_questions.json'
        ]
        for path in files:
            question_path = os.path.join(dataroot, path)
            qs = json.load(open(question_path))
            for q in qs:
                dictionary.tokenize(q['question'], True)

    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    args = parse_args()

    dictionary_path = os.path.join(args.dataroot, 'dictionary.pkl')

    d = create_dictionary(args.dataroot, args.task)
    d.dump_to_file(dictionary_path)

    d = Dictionary.load_from_file(dictionary_path)
    emb_dim = 300
    glove_file = 'glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(args.dataroot, 'glove6b_init_%dd.npy' % emb_dim), weights)
