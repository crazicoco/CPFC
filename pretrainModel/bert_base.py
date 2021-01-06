import torch
from torch import nn
from transformers import BertModel


class bertBase(nn.Module):
    def __init__(self, bert_model, labelNums):
        super(bertBase, self).__init__()
        self.bert = bert_model
        self.labelNums = labelNums
        self.atten_layer = nn.Linear(768, 16)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.OCNLI_layer = nn.Linear(768, 16 * self.labelNums[1])
        self.OCEMOTION_layer = nn.Linear(768, 16 * self.labelNums[0])
        self.TNEWS_layer = nn.Linear(768, 16 * self.labelNums[2])

    def forward(self, input_ids, token_type_ids, attention_mask, ocemotionSerial, ocnliSerial, tnewsSerial):
        cls_emb = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :].squeeze(1)
        if ocnliSerial.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocnliSerial, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocnli_value = self.OCNLI_layer(cls_emb[ocnliSerial, :]).contiguous().view(-1, 16, 3)
            ocnli_out = torch.matmul(attention_score, ocnli_value).squeeze(1)
        else:
            ocnli_out = None
        if ocemotionSerial.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocemotionSerial, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocemotion_value = self.OCEMOTION_layer(cls_emb[ocemotionSerial, :]).contiguous().view(-1, 16, 7)
            ocemotion_out = torch.matmul(attention_score, ocemotion_value).squeeze(1)
        else:
            ocemotion_out = None
        if tnewsSerial.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[tnewsSerial, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            tnews_value = self.TNEWS_layer(cls_emb[tnewsSerial, :]).contiguous().view(-1, 16, 15)
            tnews_out = torch.matmul(attention_score, tnews_value).squeeze(1)
        else:
            tnews_out = None
        return ocemotion_out, ocnli_out, tnews_out