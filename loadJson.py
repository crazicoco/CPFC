import json
import os
import random
import torch
import math
import logging


def get_label_weight(root, preocessed_data_dir, device):
    addr = os.path.join(root, preocessed_data_dir, "labelweight.pt")
    with open(addr, "r") as f:
        labelWeight = json.load(f)
    labelWoce = torch.tensor(labelWeight['OCEMOTION'])
    labelWocn = torch.tensor(labelWeight['OCNLI'])
    labelWtne = torch.tensor(labelWeight['TNEWS'])
    if device != "-1":
        labelWoce = labelWoce.cuda()
        labelWocn = labelWocn.cuda()
        labelWtne = labelWtne.cuda()
    return labelWoce, labelWocn, labelWtne



class DataLoader():
    def __init__(self, preprocessAddress, labelName, trainDataAddress, tokenize, batch_size, max_len, device, debug=False, batch_size_change=False, dataName="train"):
        # TODO load all data and label from file
        self.labelAddress = os.path.join(preprocessAddress, labelName)
        self.dataAddress = os.path.join(preprocessAddress, trainDataAddress)
        self.batchSizeChange = batch_size_change
        self.batchSize = batch_size
        self.dataName = dataName
        self.batch_rate_nums = []
        self.max_len = max_len
        self.device = device
        self.nameList = ['OCEMOTION', "OCNLI", "TNEWS"]
        self.data = self.read_data()
        self.label = self.read_label()
        self.tokenize = tokenize
        # TODO split type dataset
        if dataName != "test":
            self.label_index()
        if dataName != "test":
            if debug:
                debug_value = 20
                self.ocemotion = {'source':self.data['OCEMOTION']['source'][:debug_value], 'label':self.data['OCEMOTION']['label'][:debug_value]}
                self.ocnli = {'source':self.data['OCNLI']['source'][:debug_value], 'source2':self.data['OCNLI']['source2'][:debug_value],'label':self.data['OCNLI']['label'][:debug_value]}
                self.tnews = {'source':self.data['TNEWS']['source'][:debug_value], 'label':self.data['TNEWS']['label'][:debug_value]}
            else:
                self.ocemotion = self.data['OCEMOTION']
                self.ocnli = self.data['OCNLI']
                self.tnews = self.data['TNEWS']
        else:
            self.ocemotion = self.data['OCEMOTION']
            self.ocnli = self.data['OCNLI']
            self.tnews = self.data['TNEWS']
        #TODO the preprocess before send into model
        self.reset()
        # self.split_batch()
        # self.tokenize_process(debug)
        # self.point = 0

    def reset(self):
        self.ocnli_ids = list(range(len(self.ocnli['source'])))
        self.ocemotion_ids = list(range(len(self.ocemotion['source'])))
        self.tnews_ids = list(range(len(self.tnews['source'])))
        if self.dataName != 'test':
            # pass
            random.shuffle(self.ocnli_ids)
            random.shuffle(self.ocemotion_ids)
            random.shuffle(self.tnews_ids)


    def label_index(self):
        #TODO label_index calculate
        labelIndex = dict()
        for name in self.nameList:
            labelIndex[name] = dict()
            for i in range(len(self.label[name])):
                labelIndex[name][self.label[name][i]] = i
            for i in range(len(self.data[name]['label'])):
                self.data[name]['label'][i] = labelIndex[name][self.data[name]['label'][i]]
        pass

    def get_batch(self):
        #TODO calculate the batch_rate_nums
        ocnli_len = len(self.ocnli_ids)
        ocemotion_len = len(self.ocemotion_ids)
        tnews_len = len(self.tnews_ids)
        total_len = ocnli_len + ocemotion_len + tnews_len
        if total_len == 0:
            return None, None, None, None, None
        elif total_len > self.batchSize:
            self.calculate_data_distribution()
            ocemotion_cur = self.ocemotion_ids[:self.batch_rate_nums[0]]
            self.ocemotion_ids = self.ocemotion_ids[self.batch_rate_nums[0]:]
            #TODO write the batch result
            ocnli_cur = self.ocnli_ids[:self.batch_rate_nums[1]]
            self.ocnli_ids = self.ocnli_ids[self.batch_rate_nums[1]:]
            tnews_cur = self.tnews_ids[:self.batch_rate_nums[2]]
            self.tnews_ids = self.tnews_ids[self.batch_rate_nums[2]:]
        else:
            ocnli_cur = self.ocnli_ids
            self.ocnli_ids = []
            ocemotion_cur = self.ocemotion_ids
            self.ocemotion_ids = []
            tnews_cur = self.tnews_ids
            self.tnews_ids = []
        max_lens, src_lens, max_four_seq = self.get_max_len_for_batch(ocemotion_cur, ocnli_cur, tnews_cur)
        max_lens += 3
        # TODO padding for ocnli task
        input_ids = []
        token_type_ids = []
        attention_mask = []
        ocnli_gold = None
        ocemotion_gold = None
        tnews_gold = None
        if len(ocemotion_cur) > 0:
            # note cls sep add give the more location
            flower = self.tokenize([self.ocemotion['source'][idx] for idx in ocemotion_cur], max_length=max_lens, add_special_tokens=True, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            ocemotion_gold = torch.tensor([self.ocemotion['label'][idx] for idx in ocemotion_cur])
        # premise
        if len(ocnli_cur) > 0:
            flower = self.tokenize([self.ocnli['source'][idx] for idx in ocnli_cur],[self.ocnli['source2'][idx] for idx in ocnli_cur], add_special_tokens=True,max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            ocnli_gold = torch.tensor([self.ocnli['label'][idx] for idx in ocnli_cur])
        # # hypothesis
        # if len(ocnli_cur) > 0:
        #     flower = self.tokenize([self.ocnli['source2'][idx] for idx in ocnli_cur],add_special_tokens=True,max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
        #     input_ids.append(flower['input_ids'])
        #     token_type_ids.append(flower['token_type_ids'])
        #     attention_mask.append(flower['attention_mask'])
        #     ocnli_gold = torch.tensor([self.ocnli['label'][idx] for idx in ocnli_cur])
        if len(tnews_cur) > 0:
            flower = self.tokenize([self.tnews['source'][idx] for idx in tnews_cur], add_special_tokens=True,
                                    max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            tnews_gold = torch.tensor([self.tnews['label'][idx] for idx in tnews_cur])
        input_ids = torch.cat(input_ids, axis=0)
        token_type_ids = torch.cat(token_type_ids, axis=0)
        attention_mask = torch.cat(attention_mask, axis=0)
        src_lens = torch.tensor(src_lens)
        start = 0
        end = len(ocemotion_cur)
        ocemotion_ids = torch.tensor([idx for idx in range(start, end)])
        start = end
        end = end + len(ocnli_cur)
        ocnli_ids = torch.tensor([idx for idx in range(start, end)])
        # start = end
        # end = end + len(ocnli_cur)
        # ocnli_pre_ids = torch.tensor([idx for idx in range(start, end)])
        # start = end
        # end = end + len(ocnli_cur)
        # ocnli_hypo_ids = torch.tensor([idx for idx in range(start, end)])
        start = end
        end = end + len(tnews_cur)
        tnews_ids = torch.tensor([idx for idx in range(start, end)])
        meta = dict()
        serial = dict()
        meta['input_ids'] = input_ids
        meta['token_type_ids'] = token_type_ids
        meta['attention_mask'] = attention_mask
        serial['ocemotionSerial'] = ocemotion_ids
        serial['ocnliSerial'] =ocnli_ids
        # serial['ocnlipreSerial'] = ocnli_pre_ids
        # serial['ocnlihypSerial'] = ocnli_hypo_ids
        serial['tnewsSerial'] = tnews_ids
        serial['src_len'] = src_lens
        # serial['max_four_seq'] = max_four_seq
        if self.device != "-1":
            meta['input_ids'] = input_ids.cuda()
            meta['token_type_ids'] = token_type_ids.cuda()
            meta['attention_mask'] = attention_mask.cuda()
            serial['ocemotionSerial'] = ocemotion_ids.cuda()
            serial['ocnliSerial'] = ocnli_ids.cuda()
            # serial['ocnlipreSerial'] = ocnli_pre_ids.cuda()
            # serial['ocnlihypSerial'] = ocnli_hypo_ids.cuda()
            serial['tnewsSerial'] = tnews_ids.cuda()
            serial['src_len'] = src_lens.cuda()
            if len(ocemotion_cur) > 0:
                ocemotion_gold = ocemotion_gold.cuda()
            if len(ocnli_cur) > 0:
                ocnli_gold = ocnli_gold.cuda()
            if len(tnews_cur) > 0:
                tnews_gold = tnews_gold.cuda()
        return meta, serial, ocemotion_gold, ocnli_gold, tnews_gold

    def load_simple_task(self):
        # TODO use the ocemotion task
        ocemotion_len = len(self.ocemotion_ids)
        total_len = ocemotion_len
        if total_len == 0:
            return None, None, None, None, None
        elif total_len > self.batchSize:
            ocemotion_cur = self.ocemotion_ids[:self.batchSize]
            self.ocemotion_ids = self.ocemotion_ids[self.batchSize:]
            # TODO write the batch result
        else:
            ocemotion_cur = self.ocemotion_ids
            self.ocemotion_ids = []
        max_lens = 0
        src_lens = []
        for idx in ocemotion_cur:
            temp = len(self.ocemotion['source'][idx])
            src_lens.append(temp)
            if temp > max_lens:
                max_lens = temp
        max_lens += 2
        # TODO padding for ocnli task
        input_ids = []
        token_type_ids = []
        attention_mask = []
        ocnli_gold = None
        ocemotion_gold = None
        tnews_gold = None
        if len(ocemotion_cur) > 0:
            # note cls sep add give the more location
            flower = self.tokenize([self.ocemotion['source'][idx] for idx in ocemotion_cur], max_length=max_lens,
                                   add_special_tokens=True, padding='max_length', return_tensors='pt', truncation=True)
            input_ids.append(flower['input_ids'])
            token_type_ids.append(flower['token_type_ids'])
            attention_mask.append(flower['attention_mask'])
            ocemotion_gold = torch.tensor([self.ocemotion['label'][idx] for idx in ocemotion_cur])
        # premise
        # if len(ocnli_cur) > 0:
        #     flower = self.tokenize([self.ocnli['source'][idx] for idx in ocnli_cur],
        #                            [self.ocnli['source2'][idx] for idx in ocnli_cur], add_special_tokens=True,
        #                            max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
        #     input_ids.append(flower['input_ids'])
        #     token_type_ids.append(flower['token_type_ids'])
        #     attention_mask.append(flower['attention_mask'])
        #     ocnli_gold = torch.tensor([self.ocnli['label'][idx] for idx in ocnli_cur])
        # # hypothesis
        # if len(ocnli_cur) > 0:
        #     flower = self.tokenize([self.ocnli['source2'][idx] for idx in ocnli_cur],add_special_tokens=True,max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
        #     input_ids.append(flower['input_ids'])
        #     token_type_ids.append(flower['token_type_ids'])
        #     attention_mask.append(flower['attention_mask'])
        #     ocnli_gold = torch.tensor([self.ocnli['label'][idx] for idx in ocnli_cur])
        # if len(tnews_cur) > 0:
        #     flower = self.tokenize([self.tnews['source'][idx] for idx in tnews_cur], add_special_tokens=True,
        #                            max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
        #     input_ids.append(flower['input_ids'])
        #     token_type_ids.append(flower['token_type_ids'])
        #     attention_mask.append(flower['attention_mask'])
        #     tnews_gold = torch.tensor([self.tnews['label'][idx] for idx in tnews_cur])
        input_ids = torch.cat(input_ids, axis=0)
        token_type_ids = torch.cat(token_type_ids, axis=0)
        attention_mask = torch.cat(attention_mask, axis=0)
        src_lens = torch.tensor(src_lens)
        start = 0
        end = len(ocemotion_cur)
        ocemotion_ids = torch.tensor([idx for idx in range(start, end)])
        # start = end
        # end = end + len(ocnli_cur)
        # ocnli_ids = torch.tensor([idx for idx in range(start, end)])
        # start = end
        # end = end + len(ocnli_cur)
        # ocnli_pre_ids = torch.tensor([idx for idx in range(start, end)])
        # start = end
        # end = end + len(ocnli_cur)
        # ocnli_hypo_ids = torch.tensor([idx for idx in range(start, end)])
        # start = end
        # end = end + len(tnews_cur)
        # tnews_ids = torch.tensor([idx for idx in range(start, end)])
        meta = dict()
        serial = dict()
        meta['input_ids'] = input_ids
        meta['token_type_ids'] = token_type_ids
        meta['attention_mask'] = attention_mask
        serial['ocemotionSerial'] = ocemotion_ids
        serial['ocnliSerial'] = None
        serial['tnewsSerial'] = None
        # serial['ocnliSerial'] = ocnli_ids
        # serial['ocnlipreSerial'] = ocnli_pre_ids
        # serial['ocnlihypSerial'] = ocnli_hypo_ids
        # serial['tnewsSerial'] = tnews_ids
        serial['src_len'] = src_lens
        # serial['max_four_seq'] = max_four_seq
        if self.device != "-1":
            meta['input_ids'] = input_ids.cuda()
            meta['token_type_ids'] = token_type_ids.cuda()
            meta['attention_mask'] = attention_mask.cuda()
            serial['ocemotionSerial'] = ocemotion_ids.cuda()
            # serial['ocnliSerial'] = ocnli_ids.cuda()
            # serial['ocnlipreSerial'] = ocnli_pre_ids.cuda()
            # serial['ocnlihypSerial'] = ocnli_hypo_ids.cuda()
            # serial['tnewsSerial'] = tnews_ids.cuda()
            serial['src_len'] = src_lens.cuda()
            if len(ocemotion_cur) > 0:
                ocemotion_gold = ocemotion_gold.cuda()
            # if len(ocnli_cur) > 0:
            #     ocnli_gold = ocnli_gold.cuda()
            # if len(tnews_cur) > 0:
            #     tnews_gold = tnews_gold.cuda()
        return meta, serial, ocemotion_gold, ocnli_gold, tnews_gold

    def get_batch_simple(self, taskName):
        """
        use for test, valid check wrong
        :param taskName:
        :return:
        """
        if "label" not in self.ocemotion.keys():
            dataset = "test"
        else:
            dataset = "valid"
        ocnli_ids = torch.tensor([])
        ocemotion_ids = torch.tensor([])
        tnews_ids = torch.tensor([])
        ocnli_gold = []
        ocemotion_gold = []
        tnews_gold = []
        ocnli_s = []
        ocemotion_s = []
        tnews_s = []
        if taskName == "OCEMOTION":
            ocemotion_len = len(self.ocemotion_ids)
            if ocemotion_len == 0:
                if dataset == "test":
                    return None, None
                else:
                    return None, None, None, None, None, None, None, None
            elif ocemotion_len >self.batchSize:
                data_cur = self.ocemotion_ids[:self.batchSize]
                self.ocemotion_ids = self.ocemotion_ids[self.batchSize:]
            else:
                data_cur = self.ocemotion_ids
                self.ocemotion_ids = []
            max_lens = 0
            src_lens = []
            for idx in data_cur:
                src_lens.append(len(self.ocemotion['source'][idx]))
                if len(self.ocemotion['source'][idx]) >max_lens:
                    max_lens = len(self.ocemotion['source'][idx])
            max_lens += 2
            flower = self.tokenize([self.ocemotion['source'][idx] for idx in data_cur], add_special_tokens=True, max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
            input_ids = flower['input_ids']
            token_type_ids = flower['token_type_ids']
            attention_mask = flower['attention_mask']
            src_lens = torch.tensor(src_lens)
            if dataset == 'valid':
                ocemotion_gold = [self.ocemotion['label'][idx] for idx in data_cur]
                ocemotion_s = [self.ocemotion['source'][idx] for idx in data_cur]
            ocemotion_ids = torch.tensor([idx for idx in range(0, len(data_cur))])
        elif taskName == "OCNLI":
            ocnli_lens = len(self.ocnli_ids)
            if ocnli_lens == 0:
                if dataset == "test":
                    return None, None
                else:
                    return None, None, None, None, None, None, None, None
            elif ocnli_lens > self.batchSize:
                data_cur = self.ocnli_ids[:self.batchSize]
                self.ocnli_ids = self.ocnli_ids[self.batchSize:]
            else:
                data_cur = self.ocnli_ids
                self.ocnli_ids = []
            max_lens = 0
            src_lens = []
            for idx in data_cur:
                src_lens.append(len(self.ocnli['source'][idx]) + len(self.ocnli['source2'][idx]))
                if len(self.ocnli['source'][idx]) + len(self.ocnli['source2'][idx]) > max_lens:
                    max_lens = len(self.ocnli['source'][idx]) + len(self.ocnli['source2'][idx])
            max_lens += 3
            flower = self.tokenize([self.ocnli['source'][idx] for idx in data_cur], [self.ocnli['source2'][idx] for idx in data_cur], add_special_tokens=True,
                                   max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
            input_ids = flower['input_ids']
            token_type_ids = flower['token_type_ids']
            attention_mask = flower['attention_mask']
            ocnli_ids = torch.tensor([idx for idx in range(0, len(data_cur))])
            src_lens = torch.tensor(src_lens)
            if dataset == 'valid':
                ocnli_gold = [self.ocnli['label'][idx] for idx in data_cur]
                ocnli_s = [self.ocnli['source'][idx] + "\t" + self.ocnli['source2'][idx] for idx in data_cur]
        else:
            tnews_lens = len(self.tnews_ids)
            if tnews_lens == 0:
                if dataset == 'test':
                    return None, None
                else:
                    return None, None, None, None, None, None, None, None
            elif tnews_lens > self.batchSize:
                data_cur = self.tnews_ids[:self.batchSize]
                self.tnews_ids = self.tnews_ids[self.batchSize:]
            else:
                data_cur = self.tnews_ids
                self.tnews_ids = []
            max_lens = 0
            src_lens = []
            for idx in data_cur:
                src_lens.append(len(self.tnews['source'][idx]))
                if len(self.tnews['source'][idx]) > max_lens:
                    max_lens = len(self.tnews['source'][idx])
            flower = self.tokenize([self.tnews['source'][idx] for idx in data_cur], add_special_tokens=True,
                                   max_length=max_lens, padding='max_length', return_tensors='pt', truncation=True)
            input_ids = flower['input_ids']
            token_type_ids = flower['token_type_ids']
            attention_mask = flower['attention_mask']
            tnews_ids = torch.tensor([idx for idx in range(0, len(data_cur))])
            src_lens = torch.tensor(src_lens)
            if dataset == 'valid':
                tnews_gold = [self.tnews['label'][idx] for idx in data_cur]
                tnews_s = [self.tnews['source'][idx] for idx in data_cur]
        meta = dict()
        serial = dict()
        meta['input_ids'] = input_ids
        meta['token_type_ids'] = token_type_ids
        meta['attention_mask'] = attention_mask
        serial['ocemotionSerial'] = ocemotion_ids
        serial['ocnliSerial'] = ocnli_ids
        serial['tnewsSerial'] = tnews_ids
        serial['src_len'] = src_lens
        if self.device != "-1":
            meta['input_ids'] = input_ids.cuda()
            meta['token_type_ids'] = token_type_ids.cuda()
            meta['attention_mask'] = attention_mask.cuda()
            serial['ocemotionSerial'] = ocemotion_ids.cuda()
            serial['ocnliSerial'] = ocnli_ids.cuda()
            serial['tnewsSerial'] = tnews_ids.cuda()
            serial['src_len'] = src_lens.cuda()
        if dataset == 'test':
            return meta, serial
        else:
            return meta, serial, ocemotion_gold, ocnli_gold, tnews_gold, ocemotion_s, ocnli_s, tnews_s

    def get_label(self):
        return self.label

    def ocnli_padding(self, ocnli_cur):
        # sorted(ocnli_cur,)
        # according to ocnli, sort
        max_s1 = 0
        max_s2 = 0
        src_s1 = []
        src_s2 = []
        for idx in ocnli_cur:
            if max_s1 < len(self.data['OCNLI']['source'][idx]):
                max_s1 = len(self.data['OCNLI']['source'][idx])
            if max_s2 < len(self.data['OCNLI']['source2'][idx]):
                max_s2 = len(self.data['OCNLI']['source2'][idx])
        source_1 = []
        for idx in ocnli_cur:
            src_s1.append(len(self.data['OCNLI']['source'][idx]))
            s1 = self.data['OCNLI']['source'][idx] + (max_s1 - len(self.data['OCNLI']['source'][idx])) * '[PAD]'
            source_1.append(s1)
        source_2 = []
        for idx in ocnli_cur:
            src_s2.append(len(self.data['OCNLI']['source2'][idx]))
            s2 = self.data['OCNLI']['source2'][idx] + (max_s2 - len(self.data['OCNLI']['source2'][idx])) * '[PAD]'
            source_2.append(s2)
        # TODO translate the max_s1 and max_s2 to serial digital
        serial_max_s1 = list(range(max_s1))
        # note add a position for the sep in bert
        serial_max_s2 = list(range(max_s1+1, max_s1+max_s2+1))
        return source_1, source_2, serial_max_s1, serial_max_s2, src_s1, src_s2

    def get_label_nums(self):
        labelNums = []
        for name in self.nameList:
            labelNums.append(len(self.label[name]))
        return labelNums

    def get_max_len_for_batch(self, ocemotion_cur, ocnli_cur, tnews_cur):
        max_len = 0
        seq_lens = []
        max_ocnli_pre = 0
        max_ocnli_hyp = 0
        max_oce = 0
        max_tne = 0
        curs = [ocemotion_cur, ocnli_cur, ocnli_cur, tnews_cur]
        for k, cur in enumerate(curs):
            for idx in cur:
                if k==0:
                    lens = len(self.ocemotion['source'][idx])
                    if lens > max_oce:
                        max_oce = lens
                elif k==1:
                    lens = len(self.ocnli['source'][idx])
                    if lens > max_ocnli_pre:
                        max_ocnli_pre = lens
                elif k==2:
                    lens = len(self.ocnli['source2'][idx])
                    if lens > max_ocnli_hyp:
                        max_ocnli_hyp = lens
                else:
                    lens = len(self.tnews['source'][idx])
                    if lens > max_tne:
                        max_tne = lens
                if max_len < lens:
                    max_len = lens
                seq_lens.append(lens)
        return max_len, seq_lens, [max_oce, max_ocnli_pre, max_ocnli_hyp, max_tne]

    def calculate_data_distribution(self):
        #TODO calculate the rate
        self.batch_rate_nums = []
        total = len(self.ocemotion_ids) + len(self.ocnli_ids) + len(self.tnews_ids)
        self.data_rate_distribution = [len(self.ocemotion_ids) / total, len(self.ocnli_ids) / total, len(self.tnews_ids) / total]
        if self.batchSizeChange: # according to the nums to balance the data
            self.batchSize = self.calculate_batch_size()
        for i in range(3):
            self.batch_rate_nums.append(self.data_rate_distribution[i] * self.batchSize)
        differenceValue = []
        for i in range(3):
            differenceValue.append((abs(self.batch_rate_nums[i] - int(self.batch_rate_nums[i])), i))
            self.batch_rate_nums[i] = int(self.batch_rate_nums[i])
        differenceValue = sorted(differenceValue, key=lambda x: x[0], reverse=True)
        lens = self.batchSize - self.batch_rate_nums[0] - self.batch_rate_nums[1] - self.batch_rate_nums[2]
        for i in range(lens):
            self.batch_rate_nums[differenceValue[i][1]] += 1

        assert self.batch_rate_nums[0] + self.batch_rate_nums[1] + self.batch_rate_nums[2] == self.batchSize

    def read_label(self):
        with open(self.labelAddress, "r") as f:
            label = json.load(f)
        for e in label:
            label[e] = sorted(label[e])

        return label

    def read_data(self):
        with open(self.dataAddress, "r") as f:
            data = json.load(f)
        return data

    def calculate_min_count(self, a, b, c):
        if a < b and a < c:
            return a
        elif a > b and b < c:
            return b
        elif a > c and c < b:
            return c

    def calculate_batch_size(self):
        #TODO calculate appropriate batch_size
        sum_total = 5
        record = 0
        for i in range(10, 1000):
            sum = 0
            for j in range(3):
                value = self.data_rate_distribution[j] * i
                difference = value - int(value)
                if difference > 0.5:
                    sum += 1 - difference
                elif difference <= 0.5:
                    sum += difference - 0
            if sum < sum_total:
                sum_total = sum
                record = i
        return record



import argparse
from configTrainPredict import configTrain
from transformers import AutoTokenizer

def test():
    parser = argparse.ArgumentParser(description='test.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    configTrain(parser)
    args = parser.parse_args()
    if not os.path.exists(args.saveModelAddress):
        os.mkdir(args.saveModelAddress)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizeModel)
    trainDataLoader = DataLoader(args.processedDataDir, args.saveLabelIdName, args.saveTrainIdName, tokenizer,
                                 args.batch_size, args.max_len, args.device, debug=args.debug, batch_size_change=False)
    while(True):
        metaInput, serial, labelOcemotion, labelOcnli, labelTnews = trainDataLoader.get_batch()
        if metaInput == None:
            break

if __name__ == "__main__":
    test()