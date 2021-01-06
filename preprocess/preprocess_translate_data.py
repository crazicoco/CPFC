#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Organization  : Southeast University
# @Author        : crazicoco
# @Time          : 2021/1/2 21:27
# @Function      : word2vec + sif do the similarity for translate data and original data

import os
import torch
from gensim.models.word2vec import Word2Vec
import jieba
from collections import defaultdict
# get from data_preprocess.py
def read_special_dataset_test(address, mode=2):
    rawSource = []
    rawSource2 = []

    with open(address, "r") as f:
        rawData = f.readlines()
        for idx, data in enumerate(rawData):
            try:
                if mode == 2:
                    _, source = data.split("\t")
                    source = source.split("\n")[0]
                    rawSource.append(source)
                else:
                    _, source, source2 = data.split("\t")
                    source2 = source2.split("\n")[0]
                    rawSource2.append(source2)
                    rawSource.append(source)
            except ValueError:
                print(idx)
    if mode == 2:
        rawdata = {"source": rawSource}
    else:
        rawdata = {"source": rawSource, "source2": rawSource2}
        assert len(rawdata['source']) == len(rawdata['source2'])
    return rawdata


def read_special_dataset_train(address, mode):
    rawSource = []
    rawSource2 = []
    rawLabel = []
    if mode == 3:
        rawSource2 = []
    with open(address, "r", encoding='utf-8') as f:
        totalData = f.readlines()
        for data in totalData:
            if mode == 2:
                _, source, label = data.split("\t")
            else:
                _, source, source2, label = data.split("\t")
                rawSource2.append(source2)
            label = label.split("\n")[0]
            rawLabel.append(label)
            rawSource.append(source)
    if mode == 2:
        rawdata = {"source": rawSource, "label": rawLabel}
    else:
        rawdata = {"source": rawSource, "source2": rawSource2, "label": rawLabel}
    return rawdata

def read_stopword():
    root_dir = "/share/home/crazicoco/competition/CPFC/stopwords"
    names = ['baidu_stopwords.txt', 'cn_stopwords.txt', 'hit_stopwords.txt', 'scu_stopwords.txt']
    stopwords = defaultdict()
    for name in names:
        words = []
        with open(root_dir +name, "r", encoding='utf-8') as f:
            words = f.readlines()
            words = [words[i].split('\n')[0] for i in range(len(words))]
            stopwords[name] = words
    return stopwords


def get_tokenize(data):
    # process the simple sentence
    # use the cn_stopwords.txt
    s1_list = []
    cn_stopwords = read_stopword()['cn_stopwords.txt']
    for idx, s1 in enumerate(data['source']):
        temp = jieba.cut(s1)
        no_stword = [keyphrase for keyphrase in temp if keyphrase not in cn_stopwords]
        s1_list.append(no_stword)
    if len(data) == 3:
        s2_list = []
        for idx, s2 in enumerate(data['source2']):
            s2_list.append(jieba.cut(s2))
        return s1_list, s2_list
    return s1_list, None


def main():
    # TODO read the original data and translate data
    root_dir = "/share/home/crazicoco/competition/CPFC"
    ocemotion_original_addr = os.path.join(root_dir, 'rawData', 'OCEMOTION_train.csv')
    ocemotion_translate_addr = os.path.join(root_dir, 'rawData', 'translation','chinese','OCEMOTION_train.csv')
    ocnli_original_addr = os.path.join(root_dir, 'rawData', 'OCNLI_train.csv')
    ocnli_translate_addr = os.path.join(root_dir, 'rawData', 'translation','chinese','OCNLI_train.csv')
    tnews_original_addr = os.path.join(root_dir, 'rawData', 'TNEWS_train.csv')
    tnews_translate_addr = os.path.join(root_dir, 'rawData', 'translation','chinese','TNEWS_train.csv')
    ocemotion_original = read_special_dataset_train(ocemotion_original_addr, mode=2)
    ocemotion_translate = read_special_dataset_test(ocemotion_translate_addr, mode=2)
    ocnli_original = read_special_dataset_train(ocnli_original_addr, mode=3)
    ocnli_translate = read_special_dataset_test(ocnli_translate_addr, mode=3)
    tnews_original = read_special_dataset_train(tnews_original_addr, mode=2)
    tnews_translate = read_special_dataset_test(tnews_translate_addr, mode=2)
    ocemotion_translate['label'] = ocemotion_original['label']
    ocnli_translate['label'] = ocnli_original['label']
    tnews_translate['label'] = tnews_original['label']
    print("the data is read over")

    # TODO train word2vec model and get word embedding
    # achieve the word tokenize







if __name__ == "__main__":
    main()