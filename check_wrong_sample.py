#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Organization  : Southeast University
# @Author        : crazicoco
# @Time          : 2020/12/28 17:23
# @Function      : 检查模型处理过后的数据集有没有某些共性的数据问题，可以挑出来，然后考虑
import torch
import argparse
from configTrainPredict import configCheckWrong
from transformers import AutoTokenizer, BertTokenizer
from loadJson import DataLoader
import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
import logging
import json
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

def preCalculateSerial(preOcemotion, preOcnli, preTnews):
    if preOcemotion != None:
        preOcemotion = torch.argmax(preOcemotion, axis=1)
    if preOcnli != None:
        preOcnli = torch.argmax(preOcnli, axis=1)
    if preTnews != None:
        preTnews = torch.argmax(preTnews, axis=1)
    return preOcemotion, preOcnli, preTnews

def preSoftmax(preOcemotion, preOcnli, preTnews):
    Softmax = torch.nn.Softmax(dim=1).cuda()
    if preOcemotion != None:
        preOcemotion = Softmax(preOcemotion)
    if preOcnli != None:
        preOcnli = Softmax(preOcnli)
    if preTnews != None:
        preTnews = Softmax(preTnews)
    return preOcemotion, preOcnli, preTnews

def inference_wapper(args):
    if args.tokenizeModel == "hfl/chinese-electra-180g-large-discriminator":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizeModel)
    elif args.tokenizeModel == "hfl/chinese-electra-180g-small-discriminator":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizeModel)
    elif args.tokenizeModel == "hfl/chinese-roberta-wwm-ext":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizeModel)
    elif args.tokenizeModel == "bert-base-chinese":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizeModel)
    elif args.tokenizeModel == "hfl/chinese-roberta-wwm-ext-large":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizeModel)
    else:
        tokenizer = None
    # TODO read the dataset from test.pt and model
    record_dir = os.path.join(args.root, args.record_analysis, args.record_dir)
    devCheck = os.path.join(record_dir, args.devCheck)
    if not os.path.exists(devCheck):
        os.mkdir(devCheck)
    model_dir = os.path.join(record_dir, args.saveModel)
    pretrainModelName = os.path.join(model_dir, args.pretrainModelName)
    classifierModelName = os.path.join(model_dir, args.classifierName)
    preModel = torch.load(pretrainModelName)
    clasModel = torch.load(classifierModelName)
    if args.device != "-1":
        preModel = preModel.cuda()
        clasModel = clasModel.cuda()
    if args.device != '-1':
        Softmax = torch.nn.Softmax(dim=1).cuda()
    else:
        Softmax = torch.nn.Softmax(dim=1)
    Data = DataLoader(args.processedDataDir, args.saveLabelIdName, args.saveDataIdName, tokenizer, args.batch_size, args.max_len, args.device, debug=args.debug, batch_size_change=False, dataName=args.dataset)
    with torch.no_grad():
        print("*********************************************start valid model**************************************")
        preModel.eval()
        clasModel.eval()
        preOcemotionList = []
        preOcnliList = []
        preTnewsList = []
        label_oceList = []
        label_ocnList = []
        label_tnewsList = []
        s_oceList = []
        s_ocnList = []
        s_tnewsList = []
        label_dict = Data.get_label()
        with torch.no_grad():
            for name in ["OCEMOTION", "OCNLI", "TNEWS"]:
                while (True):
                    metaInput, serial, label_oce, label_ocn, label_tnews, s_oce, s_ocn, s_tnews = Data.get_batch_simple(name)
                    if metaInput == None:
                        break
                    output = preModel(**metaInput)
                    serial['output'] = output
                    preOcemotion, preOcnli, preTnews = clasModel(**serial)
                    # preOcemotion = None
                    # preOcnli = None
                    # preTnews = None
                    # if name == "OCEMOTION":
                    #     preOcemotion = torch.tensor([[-0.12, 0.1, 0.23], [0.5, -0.09, .0111]])
                    # elif name == 'OCNLI':
                    #     preOcnli = torch.tensor([[-0.12, 0.1, 0.23], [0.5, -0.09, .0111]])
                    # else:
                    #     preTnews = torch.tensor([[-0.12, 0.1, 0.23], [0.5, -0.09, .0111]])
                    if args.dataset == 'valid':
                        if label_oce != None:
                            label_oceList += np.array(label_oce).tolist()
                            s_oceList += s_oce
                        if label_ocn != None:
                            label_ocnList += np.array(label_ocn).tolist()
                            s_ocnList += s_ocn
                        if label_tnews != None:
                            label_tnewsList += np.array(label_tnews).tolist()
                            s_tnewsList += s_tnews
                        preOcemotion, preOcnli, preTnews = preCalculateSerial(preOcemotion, preOcnli, preTnews)
                        if args.device != "-1":
                            if preOcemotion != None:
                                preOcemotionList += np.array(preOcemotion.cpu()).tolist()
                            if preOcnli != None:
                                preOcnliList += np.array(preOcnli.cpu()).tolist()
                            if preTnews != None:
                                preTnewsList += np.array(preTnews.cpu()).tolist()
                        else:
                            if preOcemotion != None:
                                preOcemotionList += np.array(preOcemotion).tolist()
                            if preOcnli != None:
                                preOcnliList += np.array(preOcnli).tolist()
                            if preTnews != None:
                                preTnewsList += np.array(preTnews).tolist()
                        if name == "OCEMOTION":
                            pre_final = preOcemotionList
                            label_final = label_oceList
                            s_final = s_oceList
                        elif name == "OCNLI":
                            pre_final = preOcnliList
                            label_final = label_ocnList
                            s_final = s_ocnList
                        elif name == "TNEWS":
                            pre_final = preTnewsList
                            label_final = label_tnewsList
                            s_final = s_tnewsList
                        CM = confusion_matrix(label_final, pre_final)
                        # find the wrong sample
                        samples = {}
                        for i in range(CM.shape[0]):
                            samples[i] = dict()
                            for j in range(CM.shape[1]):
                                samples[i][j] = []
                        for idx in range(len(pre_final)):
                            samples[label_final[idx]][pre_final[idx]].append(s_final[idx])
                        dataset_dir = os.path.join(devCheck, name)
                        if not os.path.exists(dataset_dir):
                            os.mkdir(dataset_dir)
                        for i in range(CM.shape[0]):
                            # 创建指定label目录
                            temp_dir = os.path.join(dataset_dir, label_dict[name][i])
                            if not os.path.exists(temp_dir):
                                os.mkdir(temp_dir)
                            for j in range(CM.shape[1]):
                                temp = os.path.join(temp_dir, label_dict[name][j])
                                with open(temp, "w", encoding='utf-8') as f:
                                    for sample in samples[i][j]:
                                        f.write(sample)
                                        f.write('\n')
                    elif args.dataset == 'train':
                        preOcemotion, preOcnli, preTnews = preCalculateSerial(preOcemotion, preOcnli, preTnews)
                        if args.device != "-1":
                            if preOcemotion != None:
                                preOcemotionList += np.array(preOcemotion.cpu()).tolist()
                            if preOcnli != None:
                                preOcnliList += np.array(preOcnli.cpu()).tolist()
                            if preTnews != None:
                                preTnewsList += np.array(preTnews.cpu()).tolist()
                        else:
                            if preOcemotion != None:
                                preOcemotionList += np.array(preOcemotion).tolist()
                            if preOcnli != None:
                                preOcnliList += np.array(preOcnli).tolist()
                            if preTnews != None:
                                preTnewsList += np.array(preTnews).tolist()
                        if preOcemotion != None:
                            # preOcemotion = Softmax(preOcemotion)
                            # preOcemotionList.extend(preOcemotion)
                            label_oceList.extend(label_oce)
                            s_oceList += s_oce
                        if preOcnli != None:
                            # preOcnli = Softmax(preOcnli)
                            # preOcnliList.append(preOcnli)
                            label_ocnList.extend(label_ocn)
                            s_ocnList += s_ocn
                        if preTnews != None:
                            # preTnews = Softmax(preTnews)
                            # preTnewsList.append(preTnews)
                            label_tnewsList.extend(label_tnews)
                            s_tnewsList += s_tnews
                print("the simple task:{} is over".format(name))
            print("the check is over")
            if args.dataset == 'train':
                return preOcemotionList, preOcnliList, preTnewsList, label_oceList, label_ocnList, label_tnewsList, \
                s_oceList, s_ocnList, s_tnewsList
                # return torch.cat(preOcemotionList, axis=0), torch.cat(preOcnliList, axis=0), torch.cat(preTnewsList, axis=0), \
                #        torch.tensor(label_oceList), torch.tensor(label_ocnList), torch.tensor(label_tnewsList)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train1.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    configCheckWrong(parser)
    args = parser.parse_args()
    inference_wapper(args)