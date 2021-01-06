import torch
from transformers import RobertaTokenizer, RobertaModel

class roberta(torch.nn.Module):
    def __init__(self, preTrainModel):
        super(roberta, self).__init__()
        self.preTrainModel = preTrainModel


    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.preTrainModel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return output


class myRobertaClasifier(torch.nn.Module):

    def __init__(self, labelNums,device):
        super(myRobertaClasifier, self).__init__()
        self.labelNums = labelNums
        self.device = device
        self.atten_layer = torch.nn.Linear(1024, 16)
        self.embedding = torch.nn.Linear(1024, 150)
        # self.clsaten = torch.nn.Linear(768, 150)
        self.softmax_d1 = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(0.2)
        self.OCNLI_layer = torch.nn.Linear(1024, 16 * self.labelNums[1])
        self.OCEMOTION_layer = torch.nn.Linear(1024, 16 * self.labelNums[0])
        self.TNEWS_layer = torch.nn.Linear(1024, 16 * self.labelNums[2])
        # self.TNEWS_layer = torch.nn.Linear(300, 16 * self.labelNums[2])
        self.rnn = torch.nn.LSTM(input_size=150, hidden_size=150, bidirectional=True, num_layers=2, batch_first=True, dropout=0.1)
        # self.init_weight()
        # TODO for ocnli task
        self.clsaten = torch.nn.Linear(768, 150)
        self.premise_encoder = torch.nn.LSTM(input_size=150, hidden_size=150, bidirectional=False, num_layers=1, batch_first=True, dropout=0.1)
        self.hypothe_decoder = torch.nn.LSTM(input_size=150, hidden_size=150, bidirectional=False, num_layers=1, batch_first=True, dropout=0.1)
        self.Lear1= torch.nn.Linear(150,150)
        self.Lear2 = torch.nn.Linear(150,150)
        self.Lear3 = torch.nn.Linear(150,1)
        self.Lear4 = torch.nn.Linear(150,150)
        self.Lear5= torch.nn.Linear(150,150)
        self.Softmax = torch.nn.Softmax(dim=1)
        self.Lear_to_class = torch.nn.Linear(150, labelNums[1])

        # 去掉attention层
        self.OCNLI_layer_ = torch.nn.Linear(1024, self.labelNums[1])
        self.OCEMOTION_layer_ = torch.nn.Linear(1024, self.labelNums[0])
        self.TNEWS_layer_ = torch.nn.Linear(1024, self.labelNums[2])

        # add the common layer
        # self.common_layer = torch.nn.Sequential(torch.nn.Linear(1024, 300), torch.nn.ReLU(), torch.nn.Dropout())
        # self.OCNLI_layer_ = torch.nn.Linear(300, self.labelNums[1])
        # self.OCEMOTION_layer_ = torch.nn.Linear(300, self.labelNums[0])
        # self.TNEWS_layer_ = torch.nn.Linear(300, self.labelNums[2])

    def forward(self, output, ocemotionSerial, ocnliSerial, tnewsSerial, src_len):
        if ocnliSerial.size()[0] > 0:
        # if ocnliSerial != None:
            # TODO try to use the LSTM + attention
            # last_hidden_state = output[0][:,0:-1,:]
            # premise = self.clsaten(last_hidden_state[ocnlipreSerial, :, :])
            # hypothesis = self.clsaten(last_hidden_state[ocnlihypSerial, :, :])
            # #premise rnn
            # packed_input_src = torch.nn.utils.rnn.pack_padded_sequence(premise, src_len[ocnlipreSerial].tolist(), batch_first=True,
            #                                                            enforce_sorted=False)
            # premise_output, last_state = self.premise_encoder(packed_input_src)
            # premise_output, _ = torch.nn.utils.rnn.pad_packed_sequence(premise_output, batch_first=True)
            # # hypothe rnn
            # # init the init_state by last_state
            # h_0 = torch.zeros(1, max_four_seq[1], 150)
            # c_0 = last_state[1]
            # if self.device != '-1':
            #     h_0 = h_0.cuda()
            # packed_input_src = torch.nn.utils.rnn.pack_padded_sequence(hypothesis, src_len[ocnlihypSerial].tolist(), batch_first=True,
            #                                                            enforce_sorted=False)
            # hypothesis_output, last_state = self.premise_encoder(packed_input_src, (h_0, c_0))
            # hypothesis_output, _ = torch.nn.utils.rnn.pad_packed_sequence(hypothesis_output, batch_first=True)
            # # 获取当前所有RNN的最后输出
            # y_n = hypothesis_output[:, -1,:].unsqueeze(1)
            # temp_y = y_n.repeat(1, max_four_seq[1], 1)
            # temp_m = torch.tanh(self.Lear1(premise_output) + self.Lear2(temp_y))
            # alpha = self.Softmax(self.Lear3(temp_m))
            # premise_output = premise_output.contiguous().permute(0,2,1)
            # r = torch.matmul(premise_output, alpha).squeeze(-1)
            # y_n = y_n.contiguous().squeeze(1)
            # ocnli_temp = torch.tanh(self.Lear4(r) + self.Lear5(y_n))
            # ocnli_out = self.Softmax(self.Lear_to_class(ocnli_temp))

            # TODO origin
            # last_state = output[0][ocnliSerial,0,:]
            # pool_state = output[1][ocnliSerial,:]
            # attention_score = self.atten_layer(pool_state)
            # attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            # ocnli_value = self.OCNLI_layer(pool_state).contiguous().view(-1, 16, 3)
            # ocnli_out = torch.matmul(attention_score, ocnli_value).squeeze(1)

            # TODO 去掉attention
            last_state = output[0][ocnliSerial,0,:]
            ocnli_out =self.OCNLI_layer_(last_state)

            # TODO simple use the cls for -2 layer
            # last_2_state = output[2][-2][ocnliSerial,0,:]
            # # last_2_state = self.common_layer(last_2_state)
            # ocnli_out =self.OCNLI_layer_(last_2_state)
            # TODO use the token concat way to generate the sentence representation
        else:
            ocnli_out = None

        if ocemotionSerial.size()[0] > 0:
            # pool_state = output[1][ocemotionSerial,:]
            # last_state = output[0][ocemotionSerial,0,:]
            # attention_score = self.atten_layer(pool_state)
            # attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            # ocemotion_value = self.OCEMOTION_layer(pool_state).contiguous().view(-1, 16, 7)
            # ocemotion_out = torch.matmul(attention_score, ocemotion_value).squeeze(1)

            # TODO 去掉attention
            last_state = output[0][ocemotionSerial, 0, :]
            ocemotion_out = self.OCEMOTION_layer_(last_state)

            # TODO simple use the cls for -2 layer
            # last_2_state = output[2][-2][ocemotionSerial, 0, :]
            # # last_2_state = self.common_layer(last_2_state)
            # ocemotion_out = self.OCEMOTION_layer_(last_2_state)
            # TODO use the token concat way to generate the sentence representation

        else:
            ocemotion_out = None

        if tnewsSerial.size()[0] > 0:
        # if tnewsSerial != None:
            # pool_state = output[1][tnewsSerial, :]
            # # last_state = output[0][tnewsSerial,0,:]
            # attention_score = self.atten_layer(pool_state)
            # attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            # tnews_value = self.TNEWS_layer(pool_state).contiguous().view(-1, 16, 15)
            # tnews_out = torch.matmul(attention_score, tnews_value).squeeze(1)

            # TODO 去掉attention
            last_state = output[0][tnewsSerial, 0, :]
            tnews_out = self.TNEWS_layer_(last_state)

            # TODO simple use the cls for -2 layer
            # last_2_state = output[2][-2][tnewsSerial, 0, :]
            # # last_2_state = self.common_layer(last_2_state)
            # tnews_out = self.TNEWS_layer_(last_2_state)
            # TODO use the token concat way to generate the sentence representation
        else:
            tnews_out = None
        return ocemotion_out, ocnli_out, tnews_out
