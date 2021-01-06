import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,5'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, BertConfig, AutoConfig
import argparse
import torch
import numpy as np
from configTrainPredict import configTrain
from loadJson import DataLoader, get_label_weight
from pretrainModel.electra_large_parallel import preTrainElectraLarge, myElectraClassifier
from pretrainModel.electra_small_parallel import preTrainElectraSmall, myClassifierElectraSamll
from pretrainModel.roberta_large_parallel import roberta, myRobertaClasifier
from pretrainModel.roberta_base_parallel import myRobertaBaseClasifier
from lossCalculate import LossObeject
# confusion 行是是真实标签，列是预测标签
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
# 日志和可视化
import logging
from tensorboardX import SummaryWriter
from datetime import datetime


def adjust_learning_rate(base_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * (0.1 ** (epoch))

    # 也可以尝试将下游任务的学习率调高


def main(args):
    tokenizer= None
    preModel = None
    classifier = None
    config = None
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
    trainDataLoader = DataLoader(args.processedDataDir, args.saveLabelIdName, args.saveTrainIdName, tokenizer, args.batch_size, args.max_len, args.device, debug=args.debug, batch_size_change=False)
    validDataLoader = DataLoader(args.processedDataDir, args.saveLabelIdName, args.saveValidIdName, tokenizer, args.batch_size, args.max_len, args.device, debug=args.debug, batch_size_change=False, dataName="valid")
    labelNums = trainDataLoader.get_label_nums()
    if args.pretrainModel == "hfl/chinese-electra-180g-large-discriminator":
        if args.ifparallel:
            preModel = torch.nn.DataParallel(preTrainElectraLarge(AutoModel.from_pretrained(args.pretrainModel)))
        else:
            preModel = preTrainElectraLarge(AutoModel.from_pretrained(args.pretrainModel))
        classifier = myElectraClassifier(labelNums)
    elif args.pretrainModel == "hfl/chinese-electra-180g-small-discriminator":
        if args.ifparallel:
            preModel = torch.nn.DataParallel(preTrainElectraSmall(AutoModel.from_pretrained(args.pretrainModel)))
        else:
            preModel = preTrainElectraSmall(AutoModel.from_pretrained(args.pretrainModel))
        classifier = myClassifierElectraSamll(labelNums)
    elif args.pretrainModel == "bert_pretrain_model":
        # config = BertConfig(args.pretrainModel, output_hidden_states=True)
        if args.ifparallel:
            pass
            # preModel = torch.nn.DataParallel(bertBase(BertModel.from_pretrained(args.pretrainModel)))
    elif args.pretrainModel == "hfl/chinese-roberta-wwm-ext-large":
        # config = BertConfig(args.pretrainModel+'/config.json', output_hidden_states=True)
        if args.ifparallel:
            preModel = torch.nn.DataParallel(roberta(BertModel.from_pretrained(args.pretrainModel)))
        else:
            preModel = roberta(BertModel.from_pretrained(args.pretrainModel))
        classifier = myRobertaClasifier(labelNums, args.device)
    elif args.pretrainModel == "hfl/chinese-roberta-wwm-ext":
        # config = BertConfig(args.pretrainModel+'/config.json', output_hidden_states=True)
        if args.ifparallel:
            preModel = torch.nn.DataParallel(roberta(BertModel.from_pretrained(args.pretrainModel)))
        else:
            preModel = roberta(BertModel.from_pretrained(args.pretrainModel))
        classifier = myRobertaBaseClasifier(labelNums, args.device)
    if args.device != "-1":
        preModel = preModel.cuda()
        classifier = classifier.cuda()
    # set the parameter
    myParameter = [{"params": preModel.parameters()}, {"params": classifier.parameters()}]
    optimizer=torch.optim.Adam(myParameter, lr=args.lr)
    # labelWeight = get_label_weight(args.root, args.processedDataDir, args.device)
    # lossProcess = LossObeject(labelWeight, args.lossCalculateWay)
    lossProcess = LossObeject(device=args.device, lossCalculateWay=args.lossCalculateWay, label_nums=labelNums)
    best_dev_f1 = 0.0
    best_epoch = -1
    accumulate = args.accumulate
    if args.debug:
        printValue = 10
    else:
        printValue = 1000
    save_model = os.path.join(args.record_addr, "save_model")
    vision_log = os.path.join(args.record_addr, "vision_log")
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    if not os.path.exists(vision_log):
        os.makedirs(vision_log)
    writer = SummaryWriter(logdir=vision_log)
    cntTrain = 0
    cntValid = 0
    for epoch in range(args.epoch_size):
        # 更新学习率调整最优解
        adjust_learning_rate(base_lr=args.lr, optimizer=optimizer, epoch=epoch)
        # TODO start the train for model
        logging.info("*********************************************train model**************************************")
        logging.info("current epoch:%d" % (epoch))
        print("*********************************************train model**************************************")
        print("current epoch is {}".format(epoch))
        preModel.train()
        classifier.train()
        trainTotalLoss = 0
        labelOcemotionList = []
        labelOcnliList = []
        labelTnewsList = []
        preOcemotionList = []
        preOcnliList = []
        preTnewsList = []
        goodOcemotion = 0
        goodOcnli = 0
        goodTnews = 0
        trainCorrect = 0
        trainTotal = 0
        max_cnt = cntTrain
        cntTrain = 0
        a_step = args.a_step
        while(True):
            metaInput, serial, labelOcemotion, labelOcnli, labelTnews = trainDataLoader.get_batch()
            if metaInput == None:
                break
            if not accumulate:
                optimizer.zero_grad()
            output = preModel(**metaInput)
            serial['output'] = output
            preOcemotion, preOcnli, preTnews = classifier(**serial)
            tnews_kpi = 0.1 if len(preTnewsList) == 0 else goodTnews / len(preTnewsList)
            ocnli_kpi = 0.1 if len(preOcnliList) == 0 else goodOcnli / len(preOcnliList)
            ocemotion_kpi = 0.1 if len(preOcemotionList) == 0 else goodOcemotion / len(preOcemotionList)
            current_loss = lossProcess.compute_dtp(preTnews, preOcnli, preOcemotion, labelTnews, labelOcnli,
                                                   labelOcemotion, tnews_kpi, ocnli_kpi, ocemotion_kpi)
            trainTotalLoss += current_loss.item()
            current_loss.backward()
            if accumulate and (cntTrain + 1) % a_step == 0:
                optimizer.step()
                # gradient clear
                optimizer.zero_grad()
            tmp_total = 0
            preOcemotion, preOcnli, preTnews = lossProcess.preCalculateSerial(preOcemotion, preOcnli, preTnews)
            if args.device != "-1":
                if labelOcemotion != None:
                    labelOcemotion = np.array(labelOcemotion.cpu()).tolist()
                    preOcemotion = np.array(preOcemotion.cpu()).tolist()
                    tmp_total += len(labelOcemotion)
                else:
                    tmp_total += 0

                if labelOcnli !=None:
                    labelOcnli = np.array(labelOcnli.cpu()).tolist()
                    preOcnli = np.array(preOcnli.cpu()).tolist()
                    tmp_total += len(labelOcnli)
                else:
                    tmp_total += 0

                if labelTnews != None:
                    labelTnews = np.array(labelTnews.cpu()).tolist()
                    preTnews = np.array(preTnews.cpu()).tolist()
                    tmp_total += len(labelTnews)
                else:
                    tmp_total += 0
            else:
                if labelOcemotion != None:
                    labelOcemotion = np.array(labelOcemotion).tolist()
                    preOcemotion = np.array(preOcemotion).tolist()
                    tmp_total += len(labelOcemotion)
                else:
                    tmp_total += 0

                if labelOcnli != None:
                    labelOcnli = np.array(labelOcnli).tolist()
                    preOcnli = np.array(preOcnli).tolist()
                    tmp_total += len(labelOcnli)
                else:
                    tmp_total += 0

                if labelTnews != None:
                    labelTnews = np.array(labelTnews).tolist()
                    preTnews = np.array(preTnews).tolist()
                    tmp_total += len(labelTnews)
                else:
                    tmp_total += 0

            right_ocemotion = get_goodData(preOcemotion, labelOcemotion)
            right_ocnli = get_goodData(preOcnli, labelOcnli)
            right_tnews = get_goodData(preTnews, labelTnews)
            tmp_right = sum([right_ocemotion, right_ocnli, right_tnews])
            goodOcemotion += right_ocemotion
            goodOcnli += right_ocnli
            goodTnews += right_tnews
            trainCorrect += tmp_right
            trainTotal += tmp_total
            if preOcemotion != None:
                preOcemotionList += preOcemotion
                labelOcemotionList += labelOcemotion
            if preOcnli != None:
                preOcnliList += preOcnli
                labelOcnliList += labelOcnli
            if preTnews != None:
                preTnewsList += preTnews
                labelTnewsList += labelTnews
            #TODO 修正loss
            cntTrain += 1
            if (cntTrain+1) % printValue == 0:
                writer.add_scalar('scalar/train_loss',trainTotalLoss/cntTrain, epoch*max_cnt+cntTrain)
                logging.info("[ %d - th batch: train loss is %f valid total precise is %f ]" % (
                cntTrain + 1, trainTotalLoss / cntTrain, trainCorrect / trainTotal))
                print("[", cntTrain+1, "- th batch: train loss is", trainTotalLoss / cntTrain, ",train total precise is",trainCorrect/trainTotal,"]")
        if accumulate:
            optimizer.step()
        optimizer.zero_grad()
        #TODO output the F1 and report
        marcoF1ScoresOcemotion = f1_score(labelOcemotionList, preOcemotionList, average="macro")
        logging.info("The epoch:%d,marcoF1ScoresOcemotion:%f" % (epoch, marcoF1ScoresOcemotion))
        logging.info(f"{'confusion_matrix':*^80}")
        logging.info(confusion_matrix(labelOcemotionList, preOcemotionList, ))
        logging.info(f"{'classification_report':*^80}")
        logging.info(classification_report(labelOcemotionList, preOcemotionList, ))

        marcoF1ScoresOcnli = f1_score(labelOcnliList, preOcnliList, average="macro")
        logging.info("The epoch:%d,marcoF1ScoresOcnli:%f" % (epoch, marcoF1ScoresOcnli))
        logging.info(f"{'confusion_matrix':*^80}")
        logging.info(confusion_matrix(labelOcnliList, preOcnliList, ))
        logging.info(f"{'classification_report':*^80}")
        logging.info(classification_report(labelOcnliList, preOcnliList, ))

        marcoF1ScoresTnews = f1_score(labelTnewsList, preTnewsList, average="macro")
        logging.info("The epoch:%d,marcoF1ScoresTnews:%f" % (epoch, marcoF1ScoresTnews))
        logging.info(f"{'confusion_matrix':*^80}")
        logging.info(confusion_matrix(labelTnewsList, preTnewsList, ))
        logging.info(f"{'classification_report':*^80}")
        logging.info(classification_report(labelTnewsList, preTnewsList, ))
        marcoF1ScoresTotal = (marcoF1ScoresOcemotion +marcoF1ScoresOcnli +marcoF1ScoresTnews) / 3
        print("The epoch:{},marcoF1ScoresTotal:{}".format(epoch, marcoF1ScoresTotal))
        logging.info("The epoch:%d,marcoF1ScoresTotal:%f" % (epoch, marcoF1ScoresTotal))
        trainDataLoader.reset()

        # TODO design the validData
        logging.info("*********************************************start valid model**************************************")
        print("*********************************************start valid model**************************************")
        preModel.eval()
        classifier.eval()
        validTotalLoss = 0
        labelOcemotionList = []
        labelOcnliList = []
        labelTnewsList = []
        preOcemotionList = []
        preOcnliList = []
        preTnewsList = []
        goodOcemotion = 0
        goodOcnli = 0
        goodTnews = 0
        max_cnt_dev = cntValid
        cntValid = 0
        validCorrect = 0
        validTotal = 0
        with torch.no_grad():
            while (True):
                metaInput, serial, labelOcemotion, labelOcnli, labelTnews = validDataLoader.get_batch()
                if metaInput == None:
                    break
                output = preModel(**metaInput)
                serial['output'] = output
                preOcemotion, preOcnli, preTnews = classifier(**serial)
                tnews_kpi = 0.1 if len(preTnewsList) == 0 else goodTnews / len(preTnewsList)
                ocnli_kpi = 0.1 if len(preOcnliList) == 0 else goodOcnli / len(preOcnliList)
                ocemotion_kpi = 0.1 if len(preOcemotionList) == 0 else goodOcemotion / len(preOcemotionList)
                current_loss = lossProcess.compute_dtp(preTnews, preOcnli, preOcemotion, labelTnews, labelOcnli,
                                                       labelOcemotion, tnews_kpi, ocnli_kpi, ocemotion_kpi)
                validTotalLoss += current_loss.item()
                tmp_total = 0
                preOcemotion, preOcnli, preTnews = lossProcess.preCalculateSerial(preOcemotion, preOcnli, preTnews)
                if args.device != "-1":
                    if labelOcemotion != None:
                        labelOcemotion = np.array(labelOcemotion.cpu()).tolist()
                        preOcemotion = np.array(preOcemotion.cpu()).tolist()
                        tmp_total += len(labelOcemotion)
                    else:
                        tmp_total += 0

                    if labelOcnli != None:
                        labelOcnli = np.array(labelOcnli.cpu()).tolist()
                        preOcnli = np.array(preOcnli.cpu()).tolist()
                        tmp_total += len(labelOcnli)
                    else:
                        tmp_total += 0
                    if labelTnews != None:
                        labelTnews = np.array(labelTnews.cpu()).tolist()
                        preTnews = np.array(preTnews.cpu()).tolist()
                        tmp_total += len(labelTnews)
                    else:
                        tmp_total += 0
                else:
                    labelOcemotion = np.array(labelOcemotion).tolist()
                    preOcemotion = np.array(preOcemotion).tolist()
                    labelOcnli = np.array(labelOcnli).tolist()
                    preOcnli = np.array(preOcnli).tolist()
                    labelTnews = np.array(labelTnews).tolist()
                    preTnews = np.array(preTnews).tolist()

                right_ocemotion = get_goodData(preOcemotion, labelOcemotion)
                right_ocnli = get_goodData(preOcnli, labelOcnli)
                right_tnews = get_goodData(preTnews, labelTnews)
                tmp_right = sum([right_ocemotion, right_ocnli, right_tnews])
                goodOcemotion += right_ocemotion
                goodOcnli += right_ocnli
                goodTnews += right_tnews
                validCorrect += tmp_right
                validTotal += tmp_total
                if preOcemotion != None:
                    preOcemotionList += preOcemotion
                    labelOcemotionList += labelOcemotion
                if preOcnli != None:
                    preOcnliList += preOcnli
                    labelOcnliList += labelOcnli
                if preTnews != None:
                    preTnewsList += preTnews
                    labelTnewsList += labelTnews
                cntValid += 1
                if (cntValid + 1) % printValue == 0:
                    writer.add_scalar("scalar/valid_loss", validTotalLoss/cntValid, epoch*max_cnt_dev+cntValid)
                    logging.info("[ %d - th batch: valid loss is %f valid total precise is %f ]" % ( cntValid + 1, validTotalLoss / cntValid, validCorrect / validTotal))
                    print("[", cntValid + 1, "- th batch: valid loss is", validTotalLoss / cntValid,
                          ",valid total precise is", validCorrect/validTotal, "]")
            validDataLoader.reset()
            # TODO output the F1 and report
            marcoF1ScoresOcemotion_dev = f1_score(labelOcemotionList, preOcemotionList, average="macro")
            logging.info("The epoch:%d,marcoF1ScoresOcemotion:%f" % (epoch, marcoF1ScoresOcemotion_dev))
            logging.info(f"{'confusion_matrix':*^80}")
            logging.info(confusion_matrix(labelOcemotionList, preOcemotionList, ))
            logging.info(f"{'classification_report':*^80}")
            logging.info(classification_report(labelOcemotionList, preOcemotionList, ))

            marcoF1ScoresOcnli_dev = f1_score(labelOcnliList, preOcnliList, average="macro")
            logging.info("The epoch:%d ,marcoF1ScoresOcnli:%f" % (epoch, marcoF1ScoresOcnli_dev))
            logging.info(f"{'confusion_matrix':*^80}")
            logging.info(confusion_matrix(labelOcnliList, preOcnliList, ))
            logging.info(f"{'classification_report':*^80}")
            logging.info(classification_report(labelOcnliList, preOcnliList, ))

            marcoF1ScoresTnews_dev = f1_score(labelTnewsList, preTnewsList, average="macro")
            logging.info("The epoch:%d,marcoF1ScoresTnews:%f" % (epoch, marcoF1ScoresTnews_dev))
            logging.info(f"{'confusion_matrix':*^80}")
            logging.info(confusion_matrix(labelTnewsList, preTnewsList, ))
            logging.info(f"{'classification_report':*^80}")
            logging.info(classification_report(labelTnewsList, preTnewsList, ))

            marcoF1ScoresTotal_dev = (marcoF1ScoresOcemotion_dev + marcoF1ScoresOcnli_dev + marcoF1ScoresTnews_dev) / 3
            logging.info("The epoch:%d,marcoF1ScoresTotal:%f" % (epoch, marcoF1ScoresTotal_dev))
            print("The epoch:{},marcoF1ScoresTotal:{}".format(epoch, marcoF1ScoresTotal_dev))

            # tensorboardx
            # 可以尝试显示出所有的各个task对应的loss
            writer.add_scalars('scalar/totalF1', {"train_F1": marcoF1ScoresTotal, "valid_F1":marcoF1ScoresTotal_dev}, epoch)
            writer.add_scalars('scalar/ocemotion_F1',{'train_F1':marcoF1ScoresOcemotion, 'valid_F1':marcoF1ScoresOcemotion_dev}, epoch)
            writer.add_scalars('scalar/ocnli_F1',{'train_F1':marcoF1ScoresOcnli, 'valid_F1':marcoF1ScoresOcnli_dev}, epoch)
            writer.add_scalars('scalar/tnews_F1',{'train_F1':marcoF1ScoresTnews, 'valid_F1':marcoF1ScoresTnews_dev}, epoch)
            writer.add_scalars('scalar/total_loss', {'train_loss': trainTotalLoss, 'valid_loss': validTotalLoss}, epoch)
            writer.add_scalars('scalar/total_accuracy', {'train_acc': trainCorrect/trainTotal, 'valid_acc': validCorrect/validTotal}, epoch)
            if marcoF1ScoresTotal_dev > best_dev_f1:
                best_dev_f1 = marcoF1ScoresTotal_dev
                best_epoch = epoch
                if args.pretrainModel.find("roberta") != -1:
                    addr1 = os.path.join(save_model, "roberta_best_dev_f1_{}.pt".format(best_dev_f1))
                    addr2 = os.path.join(save_model, "classifier_best_dev_f1_{}.pt".format(best_dev_f1))
                if args.pretrainModel.find("electra") != -1:
                    addr1 = os.path.join(save_model, "electra_best_dev_f1_{}.pt".format(best_dev_f1))
                    addr2 = os.path.join(save_model, "classifier_best_dev_f1_{}.pt".format(best_dev_f1))
                torch.save(preModel, addr1)
                torch.save(classifier, addr2)
                logging.info("save the pretrainModel in %s and the classifierModel in %s" % (addr1, addr2))
            print('best epoch is:', best_epoch, '; with best f1 is:', best_dev_f1)
            logging.info("best epoch is:%d with best f1 is: %f" % (best_epoch, best_dev_f1))
    writer.close()

def get_goodData(pre, label):
    if pre !=None:
        good = 0
        for i in range(len(pre)):
            if pre[i] == label[i]:
                good += 1

        return good
    else:
        return 0



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='train1.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    configTrain(parser)
    args = parser.parse_args()
    if not os.path.exists(args.saveModelAddress):
        os.mkdir(args.saveModelAddress)
    loggingAddr = os.path.join(args.root, "record_analysis")
    if not os.path.exists(loggingAddr):
        os.mkdir(loggingAddr)
    now_time = datetime.now()
    # the file to save the project logging
    args.record_addr = str(now_time.year) + "_" + str(now_time.month) + "_" +str(now_time.day) +"_" + str(now_time.hour) + "_" +str(now_time.minute)
    if args.debug:
        args.record_addr += '_' + "debug"
    log_format = "%(message)s"
    # the all record file dir
    args.record_addr = os.path.join(loggingAddr, args.record_addr)
    if not os.path.exists(args.record_addr):
        os.mkdir(args.record_addr)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.record_addr, 'train.log'), format=log_format, filemode='w')
    logging.info("Parameters:")
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in args.__dict__.items()]
    # pass
    main(args)
