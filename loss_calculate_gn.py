#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Organization  : Southeast University
# @Author        : crazicoco
# @Time          : 2020/12/26 14:39
# @Function      : calculate the kpi
import torch
class GradNorm_loss_balance():
    """
    use the GN update the parameters for share layers
    """
    def __init__(self,task_nums, lr, alpha, device):
        self.task_nums = task_nums
        self.device = device
        self.parameters = []
        self.re_clear()
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        self.grad_loss = torch.nn.L1Loss()
        self.ocemotionLoss = torch.nn.CrossEntropyLoss()
        self.ocnliLoss = torch.nn.CrossEntropyLoss()
        self.tnewsLoss = torch.nn.CrossEntropyLoss()
        self.alpha = alpha

    def re_clear(self):
        self.ocemotion_weight_loss = torch.tensor([1], dtype=torch.float)
        self.ocnli_weight_loss = torch.tensor([1], dtype=torch.float)
        self.tnews_weight_loss = torch.tensor([1], dtype=torch.float)
        if self.device != "-1":
            self.ocemotion_weight_loss = self.ocemotion_weight_loss.cuda()
            self.ocnli_weight_loss = self.ocnli_weight_loss.cuda()
            self.tnews_weight_loss = self.tnews_weight_loss.cuda()
        self.ocemotion_weight_loss.requires_grad = True
        self.ocnli_weight_loss.requires_grad = True
        self.tnews_weight_loss.requires_grad = True
        self.parameters = [self.ocemotion_weight_loss, self.ocnli_weight_loss, self.tnews_weight_loss]
        self.ocemotion_current_loss = None
        self.ocnli_current_loss = None
        self.tnews_current_loss = None
        self.current_loss = None
        self.loss_0 = None


    def update(self, share_parameters):
        GradOceW = torch.autograd.grad(self.ocemotion_current_loss, share_parameters, retain_graph=True, create_graph=True)
        GradOceN = torch.norm(GradOceW[0],2)
        GradOcnW = torch.autograd.grad(self.ocnli_current_loss, share_parameters, retain_graph=True, create_graph=True)
        GradOcnN = torch.norm(GradOcnW[0], 2)
        GradTneW = torch.autograd.grad(self.tnews_current_loss, share_parameters, retain_graph=True, create_graph=True)
        GradTneN = torch.norm(GradTneW[0], 2)
        G_avg = torch.div(torch.add(GradOceN, torch.add(GradOcnN, GradTneN)), 3)

        rate_oce = torch.div(self.ocemotion_current_loss, self.loss_0)
        rate_ocn = torch.div(self.ocnli_current_loss, self.loss_0)
        rate_tne = torch.div(self.tnews_current_loss, self.loss_0)
        rate_avg = torch.div(torch.add(rate_oce, torch.add(rate_ocn, rate_tne)), 3)

        inv_rate_oce = torch.div(rate_oce, rate_avg)
        inv_rate_ocn = torch.div(rate_ocn, rate_avg)
        inv_rate_tne = torch.div(rate_tne, rate_avg)

        temp_oce = G_avg * inv_rate_oce ** self.alpha
        temp_ocn = G_avg * inv_rate_ocn ** self.alpha
        temp_tne = G_avg * inv_rate_tne ** self.alpha

        # 防止梯度传播
        temp_oce = temp_oce.detach()
        temp_ocn = temp_ocn.detach()
        temp_tne = temp_tne.detach()

        # calculate grad for w
        self.optimizer.zero_grad()
        Lgrad = torch.add(self.grad_loss(GradOceN, temp_oce), torch.add(self.grad_loss(GradOcnN, temp_ocn),self.grad_loss(GradTneN, temp_tne)))
        Lgrad.backward()
        self.optimizer.step()

    def renormalize_w(self):
        # renormalizing the lossed weights

        coef = 3 / torch.add(self.ocemotion_weight_loss, torch.add(self.ocnli_weight_loss, self.tnews_weight_loss))
        self.parameters = [coef * self.ocemotion_weight_loss, coef * self.ocnli_weight_loss, coef * self.tnews_weight_loss]

    def preCalculateSerial(self, preOcemotion, preOcnli, preTnews):
        if preOcemotion != None:
            preOcemotion = torch.argmax(preOcemotion, axis=1)
        if preOcnli != None:
            preOcnli = torch.argmax(preOcnli, axis=1)
        if preTnews != None:
            preTnews = torch.argmax(preTnews, axis=1)
        return preOcemotion, preOcnli, preTnews

    def compute_dtp(self, tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold):
        res = 0
        if ocemotion_pred != None:
            self.ocemotion_current_loss = self.ocemotionLoss(ocemotion_pred, ocemotion_gold) * self.parameters[0]
        if ocnli_pred != None:
            self.ocnli_current_loss = self.ocnliLoss(ocnli_pred, ocnli_gold) * self.parameters[1]
        if tnews_pred != None:
            self.tnews_current_loss = self.tnewsLoss(tnews_pred, tnews_gold) * self.parameters[2]
        res = self.ocemotion_current_loss + self.ocnli_current_loss + self.tnews_current_loss
        res = torch.div(res, 3)
        if self.loss_0 == None:
            self.loss_0 = res.data
        self.current_loss = res
        return res, self.ocemotion_current_loss, self.ocnli_current_loss, self.tnews_current_loss