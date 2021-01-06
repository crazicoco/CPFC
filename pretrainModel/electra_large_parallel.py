import torch
from itertools import chain

class preTrainElectraLarge(torch.nn.Module):
    def __init__(self, preTrainModel):
        super(preTrainElectraLarge, self).__init__()
        self.preTrainModel = preTrainModel

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.preTrainModel(input_ids, token_type_ids, attention_mask)
        return output

class myElectraClassifier(torch.nn.Module):
    def __init__(self, labelNums):
        super(myElectraClassifier, self).__init__()
        self.labelNums = labelNums
        self.dropout = torch.nn.Dropout(0.1)

        # 去掉attention层
        self.OCNLI_layer_ = torch.nn.Linear(1024, self.labelNums[1])
        self.OCEMOTION_layer_ = torch.nn.Linear(1024, self.labelNums[0])
        self.TNEWS_layer_ = torch.nn.Linear(1024, self.labelNums[2])

        # the net in ocemotion
        # self.CnnFeatureLayer_1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(4,1024), stride=1)
        # self.CnnFeatureLayer_2 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,1024), stride=1)
        # self.CnnFeatureLayer_3 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(6,1024), stride=1)
        # self.Linear_classifier = torch.nn.Linear(30, self.labelNums[0])
        # self.SoftMax = torch.nn.Softmax(dim=1)

        # the net in ocnli
        # self.CnnFeatureLayer_1_oclni = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(4, 1024), stride=1)
        # self.CnnFeatureLayer_2_oclni = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 1024), stride=1)
        # self.CnnFeatureLayer_3_oclni = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(6, 1024), stride=1)
        # self.Linear_classifier_two = torch.nn.Linear(30, self.labelNums[1])
        # self.SoftMax_two = torch.nn.Softmax(dim=1)

        # the net in tnews
        # self.CnnFeatureLayer_1_tnews = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(4, 1024), stride=1)
        # self.CnnFeatureLayer_2_tnews = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 1024), stride=1)
        # self.CnnFeatureLayer_3_tnews = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(6, 1024), stride=1)
        # self.Linear_classifier_three = torch.nn.Linear(30, self.labelNums[2])
        # self.SoftMax_three = torch.nn.Softmax(dim=1)

    def forward(self, ocemotionSerial, ocnliSerial, tnewsSerial, output, src_len):
        if ocemotionSerial.size()[0] > 0:
            # ocemotionOutput = output[ocemotionSerial,:,:].unsqueeze(1)
            # ocemotionOutput_1 = self.CnnFeatureLayer_1(ocemotionOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(ocemotionOutput_1.size(-1)))
            # ocemotionOutput_1 = Pool(ocemotionOutput_1)
            # ocemotionOutput_2 = self.CnnFeatureLayer_2(ocemotionOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(ocemotionOutput_2.size(-1)))
            # ocemotionOutput_2 = Pool(ocemotionOutput_2)
            # ocemotionOutput_3 = self.CnnFeatureLayer_3(ocemotionOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(ocemotionOutput_3.size(-1)))
            # ocemotionOutput_3 = Pool(ocemotionOutput_3)
            # ocemotionOutput = torch.cat([ocemotionOutput_1, ocemotionOutput_2, ocemotionOutput_3], dim=1).squeeze(-1)
            # ocemotionOutput = self.dropout(self.Linear_classifier(ocemotionOutput))
            # ocemotionOutput = self.SoftMax(ocemotionOutput)

            # TODO 去掉attention
            last_state = output[0][ocemotionSerial, 0, :]
            ocemotionOutput = self.OCEMOTION_layer_(last_state)
        else:
            ocemotionOutput = None

        if ocnliSerial.size()[0] >0:
            # ocnliOutput = output[ocnliSerial, :, :].unsqueeze(1)
            # ocnliOutput_1 = self.CnnFeatureLayer_1_oclni(ocnliOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(ocnliOutput_1.size(-1)))
            # ocnliOutput_1 = Pool(ocnliOutput_1)
            # ocnliOutput_2 = self.CnnFeatureLayer_2_oclni(ocnliOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(ocnliOutput_2.size(-1)))
            # ocnliOutput_2 = Pool(ocnliOutput_2)
            # ocnliOutput_3 = self.CnnFeatureLayer_3(ocnliOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(ocnliOutput_3.size(-1)))
            # ocnliOutput_3 = Pool(ocnliOutput_3)
            # ocnliOutput = torch.cat([ocnliOutput_1, ocnliOutput_2, ocnliOutput_3], dim=1).squeeze(-1)
            # ocnliOutput = self.Linear_classifier_two(ocnliOutput)
            # ocnliOutput = self.SoftMax_two(ocnliOutput)

            # TODO 去掉attention
            last_state = output[0][ocnliSerial, 0, :]
            ocnliOutput = self.OCNLI_layer_(last_state)
        else:
            ocnliOutput = None

        if tnewsSerial.size()[0] > 0:
            # tnewsOutput = output[tnewsSerial, :, :].unsqueeze(1)
            # tnewsOutput_1 = self.CnnFeatureLayer_1_oclni(tnewsOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(tnewsOutput_1.size(-1)))
            # tnewsOutput_1 = Pool(tnewsOutput_1)
            # tnewsOutput_2 = self.CnnFeatureLayer_2_oclni(tnewsOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(tnewsOutput_2.size(-1)))
            # tnewsOutput_2 = Pool(tnewsOutput_2)
            # tnewsOutput_3 = self.CnnFeatureLayer_3(tnewsOutput).squeeze(-1)
            # Pool = torch.nn.MaxPool1d(kernel_size=(tnewsOutput_3.size(-1)))
            # tnewsOutput_3 = Pool(tnewsOutput_3)
            # tnewsOutput = torch.cat([tnewsOutput_1, tnewsOutput_2, tnewsOutput_3], dim=1).squeeze(-1)
            # tnewsOutput = self.Linear_classifier_three(tnewsOutput)
            # tnewsOutput = self.SoftMax_three(tnewsOutput)

            # TODO 去掉attention
            last_state = output[0][tnewsSerial, 0, :]
            tnewsOutput = self.TNEWS_layer_(last_state)
        else:
            tnewsOutput = None
        return ocemotionOutput, ocnliOutput, tnewsOutput

