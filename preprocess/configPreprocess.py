

def basicConfig(parser):
    # parser.add_argument("-root", default="/share/home/crazicoco/competition/CPFC")
    parser.add_argument("-root", default="/home/crazicoco/competition/CPFC")

    # parser.add_argument("-root", default="G:\\NLP_task\\NLP model learning\competition\\NLP中文预训练模型泛化能力挑战赛\CPFC")
    parser.add_argument("-rawDataAddress", default="rawData")
    # save address
    parser.add_argument("-processedDataaddress", default="preprocessed_data")
    parser.add_argument("-saveLabelIdName", default="label.pt")
    parser.add_argument("-saveLabelWIdName", default="labelweight.pt")
    parser.add_argument("-saveTrainIdName", default="train.pt")
    parser.add_argument("-saveValidIdName", default="valid.pt")
    parser.add_argument("-saveTestIdName", default="test.pt")
    parser.add_argument("-valid_nums", default=3000)
