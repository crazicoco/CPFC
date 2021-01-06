import os
import argparse
from configPreprocess import basicConfig
from collections import defaultdict
import json
import math
import random
import jieba
from eda import *
# from transformers import
# split dev and train
# statistic label weight and label




def split_dataset(root, dev_data_cnt=3000):
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        cnt = 0
        with open(root + '/rawData/' + e + '_train1128.csv') as f:
            with open(root + '/rawData/' + e + '_train.csv', 'w') as f_train:
                with open( root + '/rawData/' + e + '_dev.csv', 'w') as f_dev:
                    for line in f:
                        cnt += 1
                        if cnt <= dev_data_cnt:
                            f_dev.write(line)
                        else:
                            f_train.write(line)

class DataProcess():
    """
    数据读取，训练集验证集分割，没有考虑cleanlab清洗数据
    """
    def __init__(self, args):
        self.root = os.path.join(args.root, args.rawDataAddress)
        self.label = dict()
        self.rawData = dict()
        self.testData =dict()
        self.trainData = dict()
        self.validData = dict()
        self.processedDataaddress = os.path.join(args.root, args.processedDataaddress)
        self.labelSaveAddresss = os.path.join(self.processedDataaddress, args.saveLabelIdName)
        self.labelWeightSaveAddresss = os.path.join(self.processedDataaddress, args.saveLabelWIdName)
        self.trainSaveAddress = os.path.join(self.processedDataaddress, args.saveTrainIdName)
        self.validSaveAddress = os.path.join(self.processedDataaddress, args.saveValidIdName)
        self.testSaveAddress = os.path.join(self.processedDataaddress, args.saveTestIdName)
        self.stopWordAddress = os.path.join(args.root, "stopwords/")
        self.valid_nums = args.valid_nums
        self.read_stopword()
        self.read()
    def dataEda(self, data, alpha=0.05, n=8):
        """
        给出文本返回所有文本的增强结果, 同一个标签的
        :param data:
        :param label_rate:
        :param label_dict:
        :return:
        """
        aug_sentences = []
        for src in data:
            aug_sentences += eda(src, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha,
                                 p_rd=alpha, num_aug=n)
        return aug_sentences

    def read_special_dataset_test(self, address, mode=2):
        rawSource = []
        rawSource2 = []

        with open(address, "r") as f:
            rawData = f.readlines()
            for idx, data in enumerate(rawData):
                try:
                    if mode==2:
                        _, source = data.split("\t")
                        source = source.split("\n")[0]
                        rawSource.append(source)
                    else:
                        _, source, source2= data.split("\t")
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

    def read_special_dataset_train(self, address, mode):
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

    def statistic_label(self, total):
        label_set = set()
        labelCnnSet = defaultdict(int)
        for line in total['label']:
            label_set.add(line)
            labelCnnSet[line] += 1
        labelWeightSet = []
        label_set = sorted(list(label_set))
        labelWeightSet = [labelCnnSet[e] for e in label_set]
        total_weight = sum(labelWeightSet)
        for k, v in labelCnnSet.items():
            labelCnnSet[k] = v / total_weight
        labelWeightSet = [math.log(total_weight / e) for e in labelWeightSet]
        return labelWeightSet, label_set, labelCnnSet

    def read_stopword(self):
        names = ['baidu_stopwords.txt', 'cn_stopwords.txt', 'hit_stopwords.txt', 'scu_stopwords.txt']
        stopwords = defaultdict()
        for name in names:
            words = []
            with open(self.stopWordAddress +name, "r", encoding='utf-8') as f:
                words = f.readlines()
                words = [words[i].split('\n')[0] for i in range(len(words))]
                stopwords[name] = words
        self.stopWord = stopwords

    def save_process_json(self):
        # TODO save the file split into train, valid, test with json format
        # with open(self.labelSaveAddresss, "w", encoding='utf-8') as f:
        #     json.dump(self.label, f)
        # with open(self.labelWeightSaveAddresss, "w", encoding='utf-8') as f:
        #     json.dump(self.labelWeight, f)
        # with open(self.trainSaveAddress, "w", encoding='utf-8') as f:
        #     json.dump(self.trainData, f)
        # with open(self.validSaveAddress, "w", encoding='utf-8') as f:
        #     json.dump(self.validData, f)
        with open(self.testSaveAddress, "w", encoding='utf-8') as f:
            json.dump(self.testData, f)

    def read(self):
        # TODO load all address
        OCEMOTIONTestAddress = os.path.join(self.root,"ocemotion_test_B.csv")
        OCEMOTIONTrainAddress = os.path.join(self.root, "OCEMOTION_train.csv")
        OCEMOTIONValidAddress = os.path.join(self.root, "OCEMOTION_dev.csv")
        OCEMOTIONTotalAddress = os.path.join(self.root, "OCEMOTION_train1128.csv")
        OCNLITestAddress = os.path.join(self.root, "ocnli_test_B.csv")
        OCNLITrainAddress = os.path.join(self.root, "OCNLI_train.csv")
        OCNLIValidAddress = os.path.join(self.root, "OCNLI_dev.csv")
        OCNLITotalAddress = os.path.join(self.root, "OCNLI_train1128.csv")
        TNEWSTestAddress = os.path.join(self.root, "tnews_test_B.csv")
        TNEWSTotalAddress = os.path.join(self.root, "TNEWS_train1128.csv")
        TNEWSTrainAddress = os.path.join(self.root, "TNEWS_train.csv")
        TNEWSValidAddress = os.path.join(self.root, "TNEWS_dev.csv")
        # #TODO read raw data
        OCEMOTIONTrain = self.read_special_dataset_train(address=OCEMOTIONTotalAddress, mode=2)
        OCEMOTIONValid = self.read_special_dataset_train(address=OCEMOTIONValidAddress, mode=2)
        OCNLITrain = self.read_special_dataset_train(address=OCNLITotalAddress, mode=3)
        OCNLIValid = self.read_special_dataset_train(address=OCNLIValidAddress, mode=3)
        TNEWSTrain = self.read_special_dataset_train(address=TNEWSTrainAddress, mode=2)
        TNEWSValid = self.read_special_dataset_train(address=TNEWSValidAddress, mode=2)
        OCEMOTIONTotal = self.read_special_dataset_train(address=OCEMOTIONTotalAddress, mode=2)
        OCNLITotal = self.read_special_dataset_train(address=OCNLITotalAddress, mode=3)
        TNEWSTotal = self.read_special_dataset_train(address=TNEWSTotalAddress, mode=2)
        OCEMOTIONTest = self.read_special_dataset_test(address=OCEMOTIONTestAddress, mode=2)
        OCNLITest = self.read_special_dataset_test(address=OCNLITestAddress, mode=3)
        TNEWSTest = self.read_special_dataset_test(address=TNEWSTestAddress, mode=2)
        # TODO statistic all label
        labelWocemotion, labelSetocemotion, labelCntocemotion= self.statistic_label(OCEMOTIONTotal)
        labelWocnli, labelSetocnli, labelCntocnli = self.statistic_label(OCNLITotal)
        labelWtnews, labelSettnews, labelCnttnews = self.statistic_label(TNEWSTotal)

        # # TODO read the translate data
        # ocemotion_addr = "/share/home/crazicoco/competition/CPFC/rawData/translation/chinese/OCEMOTION_train.csv"
        # ocemotion_translation = self.read_special_dataset_test(ocemotion_addr, mode=2)
        # ocemotion_translation['label'] = OCEMOTIONTrain['label']
        # ocnli_addr = "/share/home/crazicoco/competition/CPFC/rawData/translation/chinese/OCNLI_train.csv"
        # ocnli_translation = self.read_special_dataset_test(ocnli_addr, mode=3)
        # ocnli_translation['label'] = OCNLITrain['label']
        # tnews_addr = "/share/home/crazicoco/competition/CPFC/rawData/translation/chinese/TNEWS_train.csv"
        # tnews_translation = self.read_special_dataset_test(tnews_addr, mode=2)
        # tnews_translation['label'] = TNEWSTrain['label']
        # translation_data = '/share/home/crazicoco/competition/CPFC/preprocessed_data/translate.pt'
        # all_data = {"OCEMOTION":ocemotion_translation, 'OCNLI': ocnli_translation, 'TNEWS':tnews_translation}
        # with open(translation_data, "w", encoding='utf-8') as f:
        #     json.dump(all_data, f)

        # TODO read the preprocessed_data/translate_check
        # translation_data_addr = '/share/home/crazicoco/competition/CPFC/preprocessed_data/translate_checked.pt'
        # with open(translation_data_addr, "r", encoding='utf-8') as f:
        #     translation_data = json.load(f)

        # TODO consider different task has different requirement I use different process
        # OCEMOTIONTrain = self.data_process_OCE(OCEMOTIONTrain, OCEMOTIONTest, translation_data['OCEMOTION'],labelCntocemotion, labelSetocemotion)
        # OCNLITrain = self.data_process_OCN(OCNLITrain, OCNLITest, translation_data['OCNLI'], labelCntocnli, labelSetocnli)
        # TNEWSTrain = self.data_process_TNE(TNEWSTrain, TNEWSTest, translation_data['TNEWS'], labelCnttnews, labelSettnews)

        # 不能随意去除非中文字符，尤其是具有语义信息的字符，会导致语句的语义信息缺失
        #  去除长度过短的数据
        # 不能eda来处理数据，这样会大量导致数据的相似性，从而导致容易过拟合

        # TODO use stopwords tables

        # self.label = {"OCEMOTION": labelSetocemotion, "OCNLI": labelSetocnli, "TNEWS": labelSettnews}
        # self.labelWeight = {"OCEMOTION": labelWocemotion, "OCNLI": labelWocnli, "TNEWS": labelWtnews}
        # self.trainData = {"OCEMOTION": OCEMOTIONTrain, "OCNLI": OCNLITrain, "TNEWS": TNEWSTrain}
        self.testData = {"OCEMOTION":OCEMOTIONTest, "OCNLI":OCNLITest, "TNEWS":TNEWSTest}
        # self.validData = {"OCEMOTION": OCEMOTIONValid, "OCNLI": OCNLIValid, "TNEWS": TNEWSValid}

    def clear(self, data):
        """
        去除噪声
        :param data:
        :return:
        """
        # TODO 去除重复词语 和非中文字符
        for idx, src in enumerate(data['source']):
            # TODO 去除非中文字符, 包括去除标点符号，特殊符号
            constr = ""
            for ch in src:
                if ch >= u'\u4e00' and ch <= u'\u9fa5':
                    if ch != " ":
                        constr += ch
            data['source'][idx] = constr
        if len(data) >=3:
            for idx, src in enumerate(data['source2']):
                # TODO 去除非中文字符
                constr = ""
                for ch in src:
                    if ch >= u'\u4e00' and ch <= u'\u9fa5':
                        if ch != " ":
                            constr += ch
                data['source2'][idx] = constr
        return data

    def data_process_OCE(self, data, test, translate_data, label_rate, label_dict):
        """
        存在严重的数据不均衡问题, 通过处理回译数据来对平衡样本不均衡问题，注意数据多不一定好
        :param data:
        :param label_rate:
        :param label_dict:
        :return:
        """
        # TODO 分别统计出不同类别的统计词典，然后对比出现问题的几个类别的词典，分析出该如何去除对应的关键词
        # label_data = defaultdict()
        # label_data_tokens = defaultdict()
        # for na in label_dict:
        #     label_data[na] = []
        #     label_data_tokens[na] = dict()
        # for idx, label in enumerate(data['label']):
        #     label_data[label].append(idx)
        #
        # # # the label word frequent has statistic over
        # # for key in label_data.keys():
        # #     for idx in label_data[key]:
        # #         s1 = data['source'][idx]
        # #         tokens = jieba.cut(s1)
        # #         for token in tokens:
        # #             try:
        # #                 label_data_tokens[key][token] += 1
        # #             except KeyError:
        # #                 label_data_tokens[key][token] = 1
        # #     label_data_tokens[key] = sorted(label_data_tokens[key].items(), key=lambda x:x[1], reverse=True)
        # #
        # temp_addr = "/share/home/crazicoco/competition/CPFC/preprocessed_data/word_frequent_statistic.pt"
        # # with open(temp_addr, "w", encoding='utf-8') as f:
        # #     json.dump(label_data_tokens, f)
        # with open(temp_addr, "r") as f:
        #     label_data_tokens_list = json.load(f)
        # for key in label_data_tokens.keys():
        #     for word in label_data_tokens_list[key]:
        #         if word[0] not in self.stopWord['cn_stopwords.txt']:
        #             label_data_tokens[key][word[0]] = word[1]
        # # 去除停用词
        # print("OK")

        # translate_data label to entity
        print("translate label")
        for idx in range(len(translate_data['label'])):
            translate_data['label'][idx] = label_dict[translate_data['label'][idx]]

        label_translate = defaultdict()
        for na in label_dict:
            label_translate[na] = []
        for idx, label in enumerate(translate_data['label']):
            label_translate[label].append(idx)
        ocemotion_translation_data = {'source':[], 'label': []}  # 删除 sadness, happiness数据后
        for key in label_translate.keys():
            if key not in ['happiness', 'sadness']:
                for idx in label_translate[key]:
                    ocemotion_translation_data['source'].append(translate_data['source'][idx])
                    ocemotion_translation_data['label'].append(translate_data['label'][idx])
        data['source'].extend(ocemotion_translation_data['source'])
        data['label'].extend(ocemotion_translation_data['label'])

        # TODO 读取情感词典数据送入网络中学习
        # sentiment_addr = "/share/home/crazicoco/competition/CPFC/sentiment_dict/sentiment"
        # sentiment = ['正面情感词语.txt', '正面评价词语.txt', '负面情感词语.txt', '负面评价词语.txt']
        # sentiment_list = self.get_sentiment(sentiment_addr, sentiment)

        # TODO 控制长度
        # oov_test = []
        # test_length_max = 0  # 最长数据
        # test_length_sum = 0
        # test_length_min = 3
        # for idx in range(len(test['source'])):
        #     length = len(test['source'][idx])
        #     if length < test_length_min:
        #         oov_test.append({'id': idx, "source": test['source'][idx]})
        #         # print(idx)
        #         # print(test['source'][idx])
        #     if len(test['source'][idx]) > test_length_max:
        #         test_length_max = len(test['source'][idx])
        #     #     print(idx)
        #     #     print(test['source'][idx])
        #     test_length_sum += len(test['source'][idx])
        # test_length_mean = test_length_sum / len(test['source'])  # 均等长度
        #
        #
        # # statistic the length
        # oov_src = []  # 极短文本serial
        # length_max = 0   # 最长数据
        # length_sum = 0
        # length_min = 3
        # for idx in range(len(data['source'])):
        #     if len(data['source'][idx]) > length_max:
        #         length_max = len(data['source'][idx])
        #     if len(data['source'][idx]) < length_min:
        #         oov_src.append(idx)
        #         # print(idx)
        #         # print(data['source'][idx])
        #         # print(data['label'][idx])
        #     length_sum += len(data['source'][idx])
        # length_mean = length_sum / len(data['label'])   # 均等长度
        # # TODO 对于语气词控制长度， 不规范的语气词替换：省略号替换，[疯了]这样结构的词语控制个数
        # # 接下来删除极短文本
        # # 控制长文本，包括删除，修改，剪断  已经在原文档中直接修改了
        # new_data = {'source': [], 'label': []}
        # for idx, (source,label) in enumerate(zip(data['source'], data['label'])):
        #     if idx not in oov_src:
        #         new_data['source'].append(source)
        #         new_data['label'].append(label)

        label_data = defaultdict()
        for na in label_dict:
            label_data[na] = []
        for idx, label in enumerate(data['label']):
            label_data[label].append(idx)

        # TODO 控制类别均衡问题
        # 过采样
        augment_sents = {'source':[],'label':[]}
        augment_keys = ['surprise']
        augment_nums = [4]
        for i, key in enumerate(augment_keys):
            label_source = self.dataEda([data['source'][idx] for idx in label_data[key]], n=augment_nums[i])
            augment_sents['source'].extend(label_source)
            augment_sents['label'].extend([key] * len(label_source))
        data['source'].extend(augment_sents['source'])
        data['label'].extend(augment_sents['label'])

        # 欠采样
        # 尝试对sadness欠采样
        new_data = {"source":[],'label':[]}
        give_up_data = label_data['sadness'][:2000]
        for idx in range(len(data['label'])):
            if idx not in give_up_data:
                new_data['source'].append(data['source'][idx])
                new_data['label'].append(data['label'][idx])
        print("the ocemotion dataset is processed over")
        return new_data


    def data_process_TNE(self, data, test, translate_data, label_rate, label_dict):
        """
        重点放在这个任务上
        数据本身存在类别不均衡的问题，并且应该考虑数据的数据质量问题。
        :param data:
        :param label_rate:
        :param label_dict:
        :return:
        """
        # TODO load the translate data for the label balance problem
        print("translate label")
        for idx in range(len(translate_data['label'])):
            translate_data['label'][idx] = label_dict[translate_data['label'][idx]]

        label_translate = defaultdict()
        for na in label_dict:
            label_translate[na] = []
        for idx, label in enumerate(translate_data['label']):
            label_translate[label].append(idx)
        # # preprocess the translate data
        # oov_test = []
        # test_length_max = 0  # 最长数据
        # test_length_sum = 0
        # test_length_min = 999
        # for idx in range(len(test['source'])):
        #     length = len(test['source'][idx])
        #     if length < test_length_min:
        #         test_length_min = length
        #         # oov_test.append({'id': idx, "source": test['source'][idx]})
        #         # print(idx)
        #         # print(test['source'][idx])
        #     if len(test['source'][idx]) > test_length_max:
        #         test_length_max = len(test['source'][idx])
        #     #     print(idx)
        #     #     print(test['source'][idx])
        #     test_length_sum += len(test['source'][idx])
        # test_length_mean = test_length_sum / len(test['source'])  # 均等长度
        #
        # # statistic the length
        # oov_src = []  # 极短文本serial
        # length_max = 0  # 最长数据
        # length_sum = 0
        # length_min = 5
        # for idx in range(len(tnews_translation['source'])):
        #     if len(tnews_translation['source'][idx]) > length_max:
        #         length_max = len(tnews_translation['source'][idx])
        #     if len(tnews_translation['source'][idx]) < length_min:
        #         # length_min = len(tnews_translation['source'][idx])
        #         oov_src.append(idx)
        #         # oov_src.append({'id': idx, "source": tnews_translation['source'][idx]})
        #         # print(idx)
        #         # print(data['source'][idx])
        #         # print(data['label'][idx])
        #     length_sum += len(data['source'][idx])
        # length_mean = length_sum / len(data['label'])  # 均等长度
        #
        # pre_data_translate = {'source': [], 'label': []}
        # for idx, (source, label) in enumerate(zip(tnews_translation['source'], tnews_translation['label'])):
        #     if idx not in oov_src:
        #         pre_data_translate['source'].append(source)
        #         pre_data_translate['label'].append(label)

        # TODO 按照类别平衡概念逐渐添加数据
        label_data = defaultdict()
        for na in label_dict:
            label_data[na] = []
        for idx, label in enumerate(data['label']):
            label_data[label].append(idx)

        not_copy_label_type = ['103', '106','107', '116']
        for label in label_translate.keys():
            if label not in not_copy_label_type:
                data['source'].extend([translate_data['source'][item] for item in label_translate[label]])
                data['label'].extend([translate_data['label'][item] for item in label_translate[label]])

        # max_lens = 6688
        # new_data = {'source':[], "label":[]}
        # new_data['source'].extend(data['source'])
        # new_data['label'].extend(data['label'])
        # for label_name in label_dict:
        #     if len(label_data[label_name]) < max_lens/2:
        #         temp_data = [data['source'][idx] for idx in label_data[label_name]]
        #         temp = int((max_lens - len(label_data[label_name]*2)) / 8)
        #         if temp > len(label_data[label_name]):
        #             from_eda = self.dataEda(temp_data, alpha=0.05, n=16)
        #         else:
        #             from_eda = self.dataEda(temp_data[:temp], alpha=0.05, n=8)
        #         new_data['source'].extend(from_eda)
        #         new_data['label'].extend([label_name]*len(from_eda))
        #         for idx in label_translate[label_name]:
        #             new_data['source'].append(pre_data_translate['source'][idx])
        #             new_data['label'].append(pre_data_translate['label'][idx])
        #     elif len(label_data[label_name]) >= max_lens:
        #         continue
        #     else:
        #         # 取数据补全至6688  随机
        #         get_lens = max_lens - len(label_data[label_name])
        #         idx = 0
        #         count = 0
        #         lens = len(label_translate[label_name])
        #         while(True):
        #             i = label_translate[label_name][idx]
        #             rand = random.random()
        #             if rand > 0.5:
        #                 new_data['source'].append(pre_data_translate['source'][i])
        #                 new_data['label'].append(pre_data_translate['label'][i])
        #                 count += 1
        #             idx = (idx + 1) % lens
        #             if count == get_lens:
        #                 break

        # tnews_translation_data = {'source': [], 'label': []}  # 删除 sadness, happiness数据后

        # for key in label_data.keys():
        #     if key not in ['happiness', 'sadness']:
        #         for idx in label_data[key]:
        #             ocemotion_translation_data['source'].append(ocemotion_translation['source'][idx])
        #             ocemotion_translation_data['label'].append(ocemotion_translation['label'][idx])

        # new_data = {'source':[], 'label':[]}
        # for idx, src in enumerate(data['source']):
        #     constr = ""
        #     for ch in src:
        #         if ch >= u'\u4e00' and ch <= u'\u9fa5':
        #             if ch != " ":
        #                 constr += ch
        #     if len(constr) != 0:
        #         new_data['source'].append(constr)
        #         new_data['label'].append(data['label'][idx])
        print("the tnews dataset is processed over")
        return data

    def data_process_OCN(self, data, test, translate_data, label_rate, label_dict):
        """
        数据集不存在数据不均衡问题， 考虑eda和回译来增加数据，保证类别均衡问题
        :param data:
        :param label_rate:
        :param label_dict:
        :return:
        """
        # TODO add the translate data

        # translate_data label to entity
        print("translate label")
        for idx in range(len(translate_data['label'])):
            translate_data['label'][idx] = label_dict[translate_data['label'][idx]]

        label_translate = defaultdict()
        for na in label_dict:
            label_translate[na] = []
        for idx, label in enumerate(translate_data['label']):
            label_translate[label].append(idx)

        label_data = defaultdict()
        for na in label_dict:
            label_data[na] = []
        for idx, label in enumerate(data['label']):
            label_data[label].append(idx)

        # 从效果来看更应该加强1号标签的数据，效果表明，尽管1号少，但是效果反而一般
        label_get = label_translate['1'][:3200]
        for idx in range(len(translate_data['label'])):
            if idx in label_get:
                data['source'].append(translate_data['source'][idx])
                data['source2'].append(translate_data['source2'][idx])
                data['label'].append(translate_data['label'][idx])


        # oov_test = []
        # test_length_max = 0  # 最长数据
        # test_length_sum = 0
        # test_length_min = 11
        # for idx in range(len(test['source'])):
        #     length = len(test['source'][idx]) + len(test['source2'][idx])
        #     if length < test_length_min:
        #         # test_length_min = length
        #         oov_test.append({'id': idx, "source": test['source'][idx], "source2":test['source2'][idx]})
        #         # print(idx)
        #         # print(test['source'][idx])
        #     if length > test_length_max:
        #         test_length_max = length
        #     #     print(idx)
        #     #     print(test['source'][idx])
        #     test_length_sum += length
        # test_length_mean = test_length_sum / len(test['source'])  # 均等长度
        #
        #
        # # TODO 控制长度
        # oov_src = []  # 极短文本serial
        # length_max = 105   # 最长数据
        # length_sum = 0
        # length_min = 12
        # for idx in range(len(data['source'])):
        #     try:
        #         length = len(data['source'][idx]) + len(data['source2'][idx])
        #         if length < length_min:
        #             # length_min = length
        #             oov_src.append(idx)
        #             # oov_src.append({'id': idx, "source": data['source'][idx], "source2": data['source2'][idx]})
        #             # print(idx)
        #             # print(test['source'][idx])
        #         if length > length_max:
        #             length_max = length
        #             # oov_src.append({'id': idx, "source": data['source'][idx], "source2": data['source2'][idx]})
        #         #     print(idx)
        #         #     print(test['source'][idx])
        #         length_sum += length
        #     except IndexError:
        #         print(idx)
        # length_mean = length_sum / len(data['label'])   # 均等长度
        #
        # new_data = {'source': [], 'source2': [], 'label': []}
        # for idx, (source, source2, label) in enumerate(zip(data['source'], data['source2'],data['label'])):
        #     if idx not in oov_src:
        #         new_data['source'].append(source)
        #         new_data['source2'].append(source2)
        #         new_data['label'].append(label)

        # label_data = defaultdict()
        # for na in label_dict:
        #     label_data[na] = []
        # for idx, label in enumerate(data['label']):
        #     label_data[label].append(idx)
        # 去除噪声
        # for name in ['source', 'source2']:
        #     for idx, src in enumerate(data[name]):
        #         # TODO 去除非中文字符, 包括去除标点符号，特殊符号
        #         constr = ""
        #         for ch in src:
        #             if ch >= u'\u4e00' and ch <= u'\u9fa5' or ch==u'\u005b' or ch==u'\u005d' or ch=='?' or ch=='!' or ch=='。' or ch==',' or ch == '、':
        #                 if ch != " ":
        #                     constr += ch
        #         data[name][idx] = constr
        # # TODO 去除停用词
        # for idx, (s1, s2) in enumerate(zip(data['source'], data['source2'])):
        #     s1 = jieba.cut(s1, cut_all=False)
        #     temp = [w for w in s1 if w not in self.stopWord['hit_stopwords.txt']]
        #     s1 = "".join(temp)
        #     s2 = jieba.cut(s2, cut_all=False)
        #     temp = [w for w in s2 if w not in self.stopWord['hit_stopwords.txt']]
        #     s2 = "".join(temp)
        #     if len(s1) != 0 and len(s2)!=0:
        #         new_data['source'].append(s1)
        #         new_data['source2'].append(s2)
        #         new_data['label'].append(data['label'][idx])
        print("the ocnli dataset is processed over")
        return data

    def get_sentiment(self, root, sentiment_names):
        """
        返回情感词
        :param root:
        :param sentiment_names:
        :return:
        """
        sentiment = []
        for name in sentiment_names:
            addr = os.path.join(root, name)
            with open(addr, 'r', encoding='utf-8') as f:
                sents = f.readlines()
                sentiment.extend(sents)
        return sentiment

    # def keep_translate_data_quality(self, original, translate):



def check_char(args):
    with open(os.path.join(args.root, "rawData/translate/english/OCEMOTION_train_1.csv"), "r") as f:
        with open(os.path.join(args.root, "rawData/OCEMOTION_train_1.csv"), "r") as f_r:
            for lina , linb in zip(f.readlines(), f_r.readlines()):
                ida = lina.split('\t')[0]
                idb = linb.split('\t')[0]
                if ida != idb:
                    print(ida, idb)
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    basicConfig(parser)
    args = parser.parse_args()
    # split_dataset(args.root, 3000)
    process = DataProcess(args)
    process.save_process_json()
    # check_char(args)