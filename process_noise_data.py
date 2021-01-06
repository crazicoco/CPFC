#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Organization  : Southeast University
# @Author        : crazicoco
# @Time          : 2020/12/28 17:23
# @Function      : 综合多个处理模型，然后标注训练数据，调整参数使得输入数据中的噪声数据得到处理
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
import argparse
import torch
import os
import numpy as np
import json
# from inference import inference_wapper
from check_wrong_sample import inference_wapper
from configTrainPredict import configCheckWrong
def get_model():
    mdoel_1 = {'record_name':'2020_12_26_0_14', 'pretrainModel':'roberta_best_dev_f1_0.6283897867192771.pt', 'classifierModel': 'classifier_best_dev_f1_0.6283897867192771.pt'}
    return mdoel_1

def main(args):
    """
    the main function
    :return:
    """
    args.dataset = 'train'
    args.saveDataIdName = '/share/home/crazicoco/competition/CPFC/preprocessed_data/translate.pt'
    # inference_wapper(args)
    model = get_model()
    preOcemotionTensor = None
    preOcnliTensor = None
    preTnewsTensor = None
    # for idx, model in enumerate(model_list):
    args.record_dir = model['record_name']
    args.pretrainModelName = model['pretrainModel']
    args.classifierName = model['classifierModel']
    preOcemotionList, preOcnliList, preTnewsList, label_oceList, label_ocnList, label_tnewsList, s_oceList, \
    s_ocnList, s_tnewsList = inference_wapper(args)

    #TODO calculate the
    translate_new_oce = {'source':[],'label':[]}
    for idx in range(len(preOcemotionList)):
        if preOcemotionList[idx] == label_oceList[idx]:
            translate_new_oce['source'].append(s_oceList[idx])
            translate_new_oce['label'].append(label_oceList[idx])
    translate_new_ocn = {'source': [], 'label': [], 'source2':[]}
    for idx in range(len(preOcnliList)):
        if preOcnliList[idx] == label_ocnList[idx]:
            temp = s_ocnList[idx].split('\t')
            translate_new_ocn['source'].append(temp[0])
            translate_new_ocn['source2'].append(temp[1])
            translate_new_ocn['label'].append(label_ocnList[idx])
    translate_new_tne = {'source': [], 'label': []}
    for idx in range(len(preTnewsList)):
        if preTnewsList[idx] == label_tnewsList[idx]:
            translate_new_tne['source'].append(s_tnewsList[idx])
            translate_new_tne['label'].append(label_tnewsList[idx])
    translate = {"OCEMOTION":translate_new_oce, "OCNLI":translate_new_ocn, "TNEWS":translate_new_tne}
    translation_data_addr = '/share/home/crazicoco/competition/CPFC/preprocessed_data/translate_checked.pt'
    with open(translation_data_addr, "w", encoding='utf-8') as f:
        json.dump(translate, f)
    # preOcemotionTensor = torch.div(preOcemotionTensor, len(model_list))
    # preOcnliTensor = torch.div(preOcnliTensor, len(model_list))
    # preTnewsTensor = torch.div(preTnewsTensor, len(model_list))
    # preOcemotionList = torch.argmax(preOcemotionTensor, dim=1)
    # preOcnliList = torch.argmax(preOcnliTensor, dim=1)
    # preTnewsList = torch.argmax(preTnewsTensor, dim=1)
    wrong_ids = []
    # mask_oce = np.array(torch.eq(preOcemotionTensor, label_oceTensor)).tolist()
    # mask_ocn = np.array(torch.eq(preOcnliTensor, label_ocnTensor)).tolist()
    # mask_tne = np.array(torch.eq(preTnewsTensor, label_tnewsTensor)).tolist()
    # wrong_oce_ids = [idx for idx in range(len(mask_oce)) if mask_oce[idx]==0]
    # wrong_ocn_ids = [idx for idx in range(len(mask_ocn)) if mask_ocn[idx]==0]
    # wrong_tne_ids = [idx for idx in range(len(mask_tne)) if mask_tne[idx]==0]

    # TODO 统计错误标签，设置阈值，去除噪声数据





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train1.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    configCheckWrong(parser)
    args = parser.parse_args()
    # args.dataset = 'train'
    # args.saveDataIdName = 'train.pt'
    main(args)