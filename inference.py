import torch
import argparse
from configTrainPredict import configInfer
from transformers import AutoTokenizer, BertTokenizer
from loadJson import DataLoader
import os
import numpy as np
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'

def preCalculateSerial(preOcemotion, preOcnli, preTnews):
    if preOcemotion != None:
        preOcemotion = torch.argmax(preOcemotion, axis=1)
    if preOcnli != None:
        preOcnli = torch.argmax(preOcnli, axis=1)
    if preTnews != None:
        preTnews = torch.argmax(preTnews, axis=1)
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
    submission_dir = os.path.join(record_dir, args.submission)
    if not os.path.exists(submission_dir):
        os.mkdir(submission_dir)
    model_dir = os.path.join(record_dir, args.saveModel)
    pretrainModelName = os.path.join(model_dir, args.pretrainModelName)
    classifierModelName = os.path.join(model_dir, args.classifierName)
    preModel = torch.load(pretrainModelName)
    clasModel = torch.load(classifierModelName)
    if args.device != "-1":
        preModel = preModel.cuda()
        clasModel = clasModel.cuda()
    testData = DataLoader(args.processedDataDir, args.saveLabelIdName, args.saveTestIdName, tokenizer, args.batch_size, args.max_len, args.device, debug=args.debug, batch_size_change=False, dataName='test')
    with torch.no_grad():
        print("*********************************************start test model**************************************")
        preModel.eval()
        clasModel.eval()
        preOcemotionList = []
        preOcnliList = []
        preTnewsList = []
        label_dict = testData.get_label()
        with torch.no_grad():
            for name in ["OCEMOTION", "OCNLI", "TNEWS"]:
                while (True):
                    # metaInput, serial = testData.get_batch_simple("OCNLI")
                    metaInput, serial = testData.get_batch_simple(name)
                    if metaInput == None:
                        break
                    output = preModel(**metaInput)
                    serial['output'] = output
                    preOcemotion, preOcnli, preTnews = clasModel(**serial)
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
                    save_addr = os.path.join(submission_dir, "ocemotion_predict.json")
                elif name == "OCNLI":
                    pre_final = preOcnliList
                    save_addr = os.path.join(submission_dir, "ocnli_predict.json")
                elif name == "TNEWS":
                    pre_final = preTnewsList
                    save_addr = os.path.join(submission_dir, "tnews_predict.json")
                with open(save_addr,"w") as f:
                    for idx, result in enumerate(pre_final):
                        single = dict()
                        single['id'] = idx+1500
                        single['label'] = label_dict[name][result]
                        print(str(idx) + '\t' + single['label'] + '\n')
                        f.write(json.dumps(single, ensure_ascii=False))
                        if idx != len(pre_final)-1:
                            f.write("\n")





















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train1.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    configInfer(parser)
    args = parser.parse_args()
    inference_wapper(args)