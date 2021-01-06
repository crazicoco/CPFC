import torch
import math
from focal_loss import FocalLoss
class LossObeject():

    def __init__(self, device='cuda:0', label_nums=None, lossCalculateWay="general", weight=True, iffocal=False):
        self.weighted = weight
        # weightOcemotion, weightOcnli, weightTnews = labelWeight
        self.lossCalculateWay = lossCalculateWay
        if label_nums == None:
            self.ocemotionLoss = torch.nn.CrossEntropyLoss()
            self.ocnliLoss = torch.nn.CrossEntropyLoss()
            self.tnewsLoss = torch.nn.CrossEntropyLoss()
        else:
            # simple task
            if len(label_nums) == 1:
                self.ocemotionLoss = torch.nn.CrossEntropyLoss(label_nums[0])
                self.ocnliLoss = torch.nn.CrossEntropyLoss()
                self.tnewsLoss = torch.nn.CrossEntropyLoss()
            else:
                self.ocemotionLoss = torch.nn.CrossEntropyLoss(label_nums[0])
                self.ocnliLoss = torch.nn.CrossEntropyLoss(label_nums[1])
                self.tnewsLoss = torch.nn.CrossEntropyLoss(label_nums[2])
        if iffocal:
            self.ocemotionLoss = FocalLoss(label_nums[0])
            self.tnewsLoss = FocalLoss(label_nums[2])
            if device != "-1":
                self.ocemotionLoss = self.ocemotionLoss.cuda()
                self.tnewsLoss = self.tnewsLoss.cuda()
        self.loss = torch.nn.CrossEntropyLoss()

    def compute_dtp(self, tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold, tnews_kpi=0.1, ocnli_kpi=0.1, ocemotion_kpi=0.1, y=0.5):
        res = 0
        if tnews_pred != None:
            res += self.tnewsLoss(tnews_pred, tnews_gold) * self._calculate_weight(tnews_kpi, y) if self.weighted else self.loss(tnews_pred, tnews_gold) * self._calculate_weight(tnews_kpi, y)
        if ocnli_pred != None:
            res += self.ocnliLoss(ocnli_pred, ocnli_gold) * self._calculate_weight(ocnli_kpi, y) if self.weighted else self.loss(ocnli_pred, ocnli_gold) * self._calculate_weight(ocnli_kpi, y)
        if ocemotion_pred != None:
            res += self.ocemotionLoss(ocemotion_pred, ocemotion_gold) * self._calculate_weight(ocemotion_kpi, y) if self.weighted else self.loss(ocemotion_pred, ocemotion_gold) * self._calculate_weight(ocemotion_kpi, y)
        return res

    # the way to calulate the simple loss
    def calculate_loss(self, labelGold, labelPre, datasetName):
        if datasetName == "OCEMOTION":
            loss = self.ocemotionLoss(labelPre, labelGold)
        elif datasetName == "OCNLI":
            loss = self.ocnliLoss(labelPre, labelGold)
        else:
            loss = self.tnewsLoss(labelPre, labelGold)
        return loss

    # the way to calculate the total loss
    def calculate_loss_total(self, lossOcetion, lossOcnli, lossTnews):
        if self.lossCalculateWay == "general":
            lossSum = 0
            if lossOcetion != None:
                lossSum = lossOcetion
            if lossOcnli != None:
                lossSum += lossOcnli
            if lossTnews != None:
                lossSum += lossTnews
            return lossSum

    def calculate_right_count(self, labelOcemotionPred, labelOcemotionGold, labelOcnliPred, labelOcnliGold, labelTnewsPred, labelTnewsGold):
        rightOcemotion = 0
        rightOcnli = 0
        rightTnews = 0
        for i in range(len(labelOcemotionGold)):
            result = torch.argmax(labelOcemotionPred[i], axis=1)
            if result == labelOcemotionGold[i]:
                rightOcemotion += 1
        for i in range(len(labelOcnliGold)):
            result = torch.argmax(labelOcnliPred[i], axis=1)
            if result == labelOcnliGold[i]:
                rightOcnli += 1
        for i in range(len(labelTnewsGold)):
            result = torch.argmax(labelTnewsPred[i], axis=1)
            if result == labelTnewsGold:
                rightTnews += 1
        return rightOcemotion, rightOcnli, rightTnews

    def preCalculateSerial(self, preOcemotion, preOcnli, preTnews):
        if preOcemotion != None:
            preOcemotion = torch.argmax(preOcemotion, axis=1)
        if preOcnli != None:
            preOcnli = torch.argmax(preOcnli, axis=1)
        if preTnews != None:
            preTnews = torch.argmax(preTnews, axis=1)
        return preOcemotion, preOcnli, preTnews


    def _calculate_weight(self, kpi, y):
        kpi = max(0.1, kpi)
        kpi = min(0.99, kpi)
        w = -1 * ((1 - kpi) ** y) * math.log(kpi)
        return w


import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
class kpi():
    def __init__(self, label_nums):
        self.appha = 0.0000001
        self.label_nums = label_nums
        self.CM = np.zeros([label_nums, label_nums], np.int64)
        pass

    def update_CM(self, pre_label, true_label):
        for idx in range(len(pre_label)):
            self.CM[true_label[idx]][pre_label[idx]] += 1

    def calculate_FN(self, idx):
        sum = self.appha
        for i in range(self.label_nums):
            sum += self.CM[idx][i]
        return sum

    def calculate_FP(self, idx):
        sum = self.appha
        for i in range(self.label_nums):
            sum += self.CM[i][idx]
        return sum

    def get_kpi(self):
        # calculate the current F1 value for simple class
        macro_f1 = 0
        for i in range(self.label_nums):
            precise_value = self.CM[i][i] / self.calculate_FP(i)
            recall = self.CM[i][i] / self.calculate_FP(i)
            sum = precise_value + recall + self.appha
            f1 = 2 * precise_value * recall / sum
            macro_f1 += f1

        return macro_f1