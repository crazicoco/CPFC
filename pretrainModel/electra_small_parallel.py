import torch
from itertools import chain

class preTrainElectraSmall(torch.nn.Module):
    def __init__(self, preTrainModel):
        super(preTrainElectraSmall, self).__init__()
        self.preTrainModel = preTrainModel

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.preTrainModel(input_ids, token_type_ids, attention_mask)
        return output[0]

class myClassifierElectraSamll(torch.nn.Module):
    def __init__(self, labelNums):
        super(myClassifierElectraSamll, self).__init__()
        self.labelNums = labelNums
        self.dropout = torch.nn.Dropout(0.1)

        # the net in ocemotion
        self.CnnFeatureLayer_1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(4,256), stride=1)
        self.CnnFeatureLayer_2 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,256), stride=1)
        self.CnnFeatureLayer_3 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(6,256), stride=1)
        self.Linear_classifier = torch.nn.Linear(30, self.labelNums[0])
        self.SoftMax = torch.nn.Softmax(dim=1)

        # the net in ocnli
        self.CnnFeatureLayer_1_oclni = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(4, 256), stride=1)
        self.CnnFeatureLayer_2_oclni = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 256), stride=1)
        self.CnnFeatureLayer_3_oclni = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(6, 256), stride=1)
        self.Linear_classifier_two = torch.nn.Linear(30, self.labelNums[1])
        self.SoftMax_two = torch.nn.Softmax(dim=1)

        # the net in tnews
        self.CnnFeatureLayer_1_tnews = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(4, 256), stride=1)
        self.CnnFeatureLayer_2_tnews = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 256), stride=1)
        self.CnnFeatureLayer_3_tnews = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(6, 256), stride=1)
        self.Linear_classifier_three = torch.nn.Linear(30, self.labelNums[2])
        self.SoftMax_three = torch.nn.Softmax(dim=1)

    def forward(self, ocemotionSerial, ocnliSerial, tnewsSerial, output):
        if ocemotionSerial.size()[0] > 0:
            ocemotionOutput = output[ocemotionSerial,:,:].unsqueeze(1)
            ocemotionOutput_1 = self.CnnFeatureLayer_1(ocemotionOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(ocemotionOutput_1.size(-1)))
            ocemotionOutput_1 = Pool(ocemotionOutput_1)
            ocemotionOutput_2 = self.CnnFeatureLayer_2(ocemotionOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(ocemotionOutput_2.size(-1)))
            ocemotionOutput_2 = Pool(ocemotionOutput_2)
            ocemotionOutput_3 = self.CnnFeatureLayer_3(ocemotionOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(ocemotionOutput_3.size(-1)))
            ocemotionOutput_3 = Pool(ocemotionOutput_3)
            ocemotionOutput = torch.cat([ocemotionOutput_1, ocemotionOutput_2, ocemotionOutput_3], dim=1).squeeze(-1)
            ocemotionOutput = self.dropout(self.Linear_classifier(ocemotionOutput))
            ocemotionOutput = self.SoftMax(ocemotionOutput)
        else:
            ocemotionOutput = None

        if ocnliSerial.size()[0] >0:
            ocnliOutput = output[ocnliSerial, :, :].unsqueeze(1)
            ocnliOutput_1 = self.CnnFeatureLayer_1_oclni(ocnliOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(ocnliOutput_1.size(-1)))
            ocnliOutput_1 = Pool(ocnliOutput_1)
            ocnliOutput_2 = self.CnnFeatureLayer_2_oclni(ocnliOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(ocnliOutput_2.size(-1)))
            ocnliOutput_2 = Pool(ocnliOutput_2)
            ocnliOutput_3 = self.CnnFeatureLayer_3(ocnliOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(ocnliOutput_3.size(-1)))
            ocnliOutput_3 = Pool(ocnliOutput_3)
            ocnliOutput = torch.cat([ocnliOutput_1, ocnliOutput_2, ocnliOutput_3], dim=1).squeeze(-1)
            ocnliOutput = self.Linear_classifier_two(ocnliOutput)
            ocnliOutput = self.SoftMax_two(ocnliOutput)
        else:
            ocnliOutput = None

        if tnewsSerial.size()[0] > 0:
            tnewsOutput = output[tnewsSerial, :, :].unsqueeze(1)
            tnewsOutput_1 = self.CnnFeatureLayer_1_oclni(tnewsOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(tnewsOutput_1.size(-1)))
            tnewsOutput_1 = Pool(tnewsOutput_1)
            tnewsOutput_2 = self.CnnFeatureLayer_2_oclni(tnewsOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(tnewsOutput_2.size(-1)))
            tnewsOutput_2 = Pool(tnewsOutput_2)
            tnewsOutput_3 = self.CnnFeatureLayer_3(tnewsOutput).squeeze(-1)
            Pool = torch.nn.MaxPool1d(kernel_size=(tnewsOutput_3.size(-1)))
            tnewsOutput_3 = Pool(tnewsOutput_3)
            tnewsOutput = torch.cat([tnewsOutput_1, tnewsOutput_2, tnewsOutput_3], dim=1).squeeze(-1)
            tnewsOutput = self.Linear_classifier_three(tnewsOutput)
            tnewsOutput = self.SoftMax_three(tnewsOutput)
        else:
            tnewsOutput = None
        return ocemotionOutput, ocnliOutput, tnewsOutput

class myDataParallel(torch.nn.DataParallel):
    """
    my DataParallel class achieve myself splices the tensor
    """
    def __init__(self, module_pretrain, label_nums):
        super(myDataParallel).__init__(module_pretrain)
        self.model_classifier = myClassifierElectraSamll(label_nums)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))
        meta = dict()
        meta['ocemotionSerial'] = kwargs['ocemotionSerial']
        meta['ocnliSerial'] = kwargs['ocnliSerial']
        meta['tnewsSerial'] = kwargs['tnewsSerial']
        parkwargs = dict()
        parkwargs['input_ids'] = kwargs['input_ids']
        parkwargs['token_type_ids'] = kwargs['token_type_ids']
        parkwargs['attention_mask'] = kwargs['attention_mask']
        #TODO achieve the scatter data

        # inputs, kwargs = self.split_data(inputs, kwargs)
        inputs, kwargs = self.scatter(inputs, parkwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        outputs = self.gather(outputs, self.output_device)
        ocemotionOutput, ocnliOutput, tnewsOutput = self.model_classifier(outputs, meta)
        return ocemotionOutput, ocnliOutput, tnewsOutput

    def split_data(self, inputs, kwargs):
        """
        split the data
        :param kwargs:
        :return:
        """
        input_ids = kwargs['input_ids']
        token_type_ids = kwargs['token_type_ids']
        attention_mask = kwargs['attention_mask']
        ocemotionSerial = kwargs['ocemotionSerial']
        ocnliSerial = kwargs['ocnliSerial']
        tnewsSerial = kwargs['tnewsSerial']
        if len(self.device_ids) == 3:
            ocemotion = {"input_ids":input_ids[ocemotionSerial,:,:], "token_type_ids":token_type_ids[ocemotionSerial,:,:], "attention_mask":attention_mask[ocemotionSerial,:,:]}
            ocnli = {"input_ids":input_ids[ocnliSerial,:,:], "token_type_ids":token_type_ids[ocnliSerial,:,:], "attention_mask":attention_mask[ocnliSerial,:,:]}
            tnews = {"input_ids":input_ids[tnewsSerial,:,:], "token_type_ids":token_type_ids[tnewsSerial,:,:], "attention_mask":attention_mask[tnewsSerial,:,:]}
            kwargs =  (ocemotion, ocnli, tnews)
            if len(inputs) == 0:
                inputs = ((), (), ())
            return inputs, kwargs



