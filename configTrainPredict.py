
def configTrain(parser):
    # TODO Address and some file name
    # parser.add_argument("-root", default="/home/crazicoco/competition/CPFC")
    parser.add_argument("-root", default="/share/home/crazicoco/competition/CPFC", type=str)
    # model
    # parser.add_argument("-pretrainModelDir", default="hfl")
    parser.add_argument("-vocabIdName", default="vocab.txt", type=str)
    parser.add_argument("-tokenizeModel", default="hfl/chinese-roberta-wwm-ext-large", type=str)
    parser.add_argument("-pretrainModel", default="hfl/chinese-roberta-wwm-ext-large", type=str)
    parser.add_argument("-saveModelAddress", default="saveModelBin/", type=str)
    # data dir
    parser.add_argument("-processedDataDir", default="preprocessed_data", type=str)
    parser.add_argument("-saveLabelIdName", default="label.pt", type=str)
    parser.add_argument("-saveTrainIdName", default="train.pt", type=str)
    parser.add_argument("-saveValidIdName", default="valid.pt", type=str)
    parser.add_argument("-saveTestIdName", default="test.pt", type=str)
    # super parameters
    parser.add_argument("-batch_size", default=9, type=int)
    parser.add_argument("-epoch_size", default=20, type=int)
    parser.add_argument("-loss_calculate", default="cross-entroy", type=str)
    parser.add_argument("-lr", default=0.00002, type=float)
    parser.add_argument("-device", default='-1', type=str)
    parser.add_argument("-max_len", default=512, type=int)
    parser.add_argument("-lossCalculateWay", default="general", type=str)
    # train
    parser.add_argument("-accumulate", default=True, type=bool)
    parser.add_argument("-ifparallel", default=False, action='store_true')
    parser.add_argument("-debug", default=False, action="store_true")
    parser.add_argument("-logfileName", default="public", type=str)
    parser.add_argument("-a_step", default=16, type=int)
    parser.add_argument("-description", default="", type=str)
    parser.add_argument("-use_bert_layer", default=-1, type=int)


def configInfer(parser):
    parser.add_argument("-root", default="/share/home/crazicoco/competition/CPFC/")
    parser.add_argument('-record_analysis', default="record_analysis")
    parser.add_argument("-record_dir", default="2020_12_26_10_5", type=str)
    parser.add_argument("-saveModel", default="save_model")
    parser.add_argument("-submission",default="submission")
    parser.add_argument("-pretrainModelName", default="roberta_best_dev_f1_0.6257867795149347.pt", type=str)
    parser.add_argument("-classifierName", default="classifier_best_dev_f1_0.6257867795149347.pt", type=str)
    parser.add_argument("-tokenizeModel", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("-saveTestIdName", default="test.pt")
    parser.add_argument("-saveLabelIdName", default="label.pt")
    parser.add_argument("-processedDataDir", default="preprocessed_data")
    parser.add_argument("-device", default='0,1,3', type=str)
    parser.add_argument("-saveTestAddr", default="0.6269")
    parser.add_argument("-batch_size", default=16)
    parser.add_argument("-max_len", default=512)
    parser.add_argument("-debug", default=False, action="store_true")


def configCheckWrong(parser):
    parser.add_argument("-root", default="/share/home/crazicoco/competition/CPFC/")
    parser.add_argument('-record_analysis', default="record_analysis")
    parser.add_argument("-record_dir", default="2020_12_25_10_52", type=str)
    parser.add_argument("-saveModel", default="save_model")
    parser.add_argument("-devCheck", default="dev_check")
    parser.add_argument("-pretrainModelName", default="roberta_best_dev_f1_0.6230825774172944.pt", type=str)
    parser.add_argument("-classifierName", default="classifier_best_dev_f1_0.6230825774172944.pt", type=str)
    parser.add_argument("-tokenizeModel", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("-saveDataIdName", default="valid.pt", type=str)
    parser.add_argument("-saveLabelIdName", default="label.pt")
    parser.add_argument("-processedDataDir", default="preprocessed_data")
    parser.add_argument("-device", default='0,1,3', type=str)
    parser.add_argument("-saveTestAddr", default="0.6269")
    parser.add_argument("-batch_size", default=16, type=int)
    parser.add_argument("-max_len", default=512)
    parser.add_argument("-debug", default=False, action="store_true")
    parser.add_argument('-dataset', default='valid', type=str)






