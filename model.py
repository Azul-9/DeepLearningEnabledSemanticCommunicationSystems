"""
it includes some basic model and function of deepSC
"""


import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def embedding(input_size, output_size):  # embedding layer, the former is the size of dic and
    # the latter is the dimension of the embedding vector
    return nn.Embedding(input_size, output_size)

def dense(input_size, output_size):  # dense layer is a full connection layer and used to gather information
    return torch.nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU()
    )

def AWGN_channel(x, snr):  # used to simulate additive white gaussian noise channel
    [batch_size, length, len_feature] = x.shape
    x_power = torch.sum(torch.abs(x)) / (batch_size * length * len_feature)
    n_power = x_power / (10 ** (snr / 10.0))
    noise = torch.rand(batch_size, length, len_feature, device=device) *n_power
    return x + noise

class SemanticCommunicationSystem(nn.Module):  # pure DeepSC
    def __init__(self):
        super(SemanticCommunicationSystem, self).__init__()
        self.embedding = embedding(35632, 128)  # which means the corpus has 35632 kinds of words and
        # each word will be coded with a 128 dimensions vector
        self.frontEncoder = nn.TransformerEncoderLayer(d_model=128, nhead=8)  # according to the paper
        self.encoder = nn.TransformerEncoder(self.frontEncoder, num_layers=3)
        self.denseEncoder1 = dense(128, 256)
        self.denseEncoder2 = dense(256, 16)

        self.denseDecoder1 = dense(16, 256)
        self.denseDecoder2 = dense(256, 128)
        self.frontDecoder = nn.TransformerDecoderLayer(d_model=128, nhead=8)
        self.decoder = nn.TransformerDecoder(self.frontDecoder, num_layers=3)

        self.prediction = nn.Linear(128, 35632)
        self.softmax = nn.Softmax(dim=2)  # dim=2 means that it calculates softmax in the feature dimension

    def forward(self, inputs):
        embeddingVector = self.embedding(inputs)
        code = self.encoder(embeddingVector)
        denseCode = self.denseEncoder1(code)
        codeSent = self.denseEncoder2(denseCode)
        codeWithNoise = AWGN_channel(codeSent, 12)  # assuming snr = 12db
        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        codeSemantic = self.decoder(codeReceived, code)
        codeSemantic = self.prediction(codeSemantic)
        info = self.softmax(codeSemantic)
        return info


class MutualInfoSystem(nn.Module):  # mutual information used to maximize channel capacity
    def __init__(self):
        super(MutualInfoSystem, self).__init__()
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # nn.init.normal_(self.fc1.weight, std=0.02)  # init weight with normal distribution and mean is 0, std is 0.02
        # nn.init.constant_(self.fc1.bias, 0)  # init bias with constant num 0
        # nn.init.normal_(self.fc2.weight, std=0.02)
        # nn.init.constant_(self.fc2.bias, 0)
        # nn.init.normal_(self.fc3.weight, std=0.02)
        # nn.init.constant_(self.fc3.bias, 0)  # which may not be necessary to initialize weight manually

    def forward(self, inputs):
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        return output

def sample_batch(batch_size, sample_mode):  # used to sample data for mutual info system
    if sample_mode == "joint":  # joint sample
        index = np.random.choice(range(12799), size=batch_size, replace=False)  # replace = false means it won't select same number
        num = 0
        for i in index:
            x = np.load("mutual data/x1/" + str(i) + ".npy")
            y = np.load("mutual data/y1/" + str(i) + ".npy")
            data_x = x.reshape(-1, 16)  # -1 means python will infer the dimension automatically
            data_y = y.reshape(-1, 16)
            if num == 0:
                batch_x = data_x
                batch_y = data_y
            else:
                batch_x = np.concatenate([batch_x, data_x], axis=0)
                batch_y = np.concatenate([batch_y, data_y], axis=0)
            num += 1
    elif sample_mode == 'marginal':  # marginal sample
        joint_index = np.random.choice(range(12799), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(12799), size=batch_size, replace=False)
        num = 0
        for i in range(batch_size):
            j_index = joint_index[i]
            m_index = marginal_index[i]
            x = np.load("mutual data/x1/" + str(j_index) + ".npy")
            y = np.load("mutual data/y1/" + str(m_index) + ".npy")
            data_x = x.reshape(-1, 16)
            data_y = y.reshape(-1, 16)
            if num == 0:
                batch_x = data_x
                batch_y = data_y
            else:
                batch_x = np.concatenate([batch_x, data_x], axis=0)
                batch_y = np.concatenate([batch_y, data_y], axis=0)
            num += 1
    batch = np.concatenate([batch_x, batch_y], axis=1)  # axis = 1 means that data will concat at the dim of row
    # and if axis = 0, which means that data will concat one by one

    return batch

class LossFn(nn.Module):  # Loss function
    def __init__(self):
        super(LossFn, self).__init__()

    def forward(self, output, label, length_sen, num_sample, batch_size):  # num_sample means the num of sentence
        # considering that num_sample may not the integer multiple of batch_size
        delta = 1e-7  # used to avoid vanishing gradient
        result = 0
        for i in range(num_sample):  # for every sentence in batch
            length = length_sen[i]  # get every length of sentence, attention that it's the length of sen without padding
            output_term = output[i, 0:length, :]  # get the sentence of corresponding vector
            label_term = label[i, 0:length, :]
            result -= (torch.sum(label_term * torch.log(output_term + delta)) / length)
        return result/batch_size

def calBLEU(n_gram, s_predicted, s, length):
    num_gram = length - n_gram + 1  # when n_gram = 1, num_gram = length, in which case the BLEU will calculate by one word
    # and the same, when n_gram = 2, num_gram = length - 1, in which case the BLEU will calculate by two words
    # so it's used to padding zero matrix
    s_predicted_gram = np.zeros((num_gram, n_gram))
    s_gram = np.zeros((num_gram, n_gram))  # used to create a matrix which stores word group to calculate matrix
    gram = np.zeros((2*num_gram, n_gram))
    count = 0
    for i in range(num_gram):
        s_predicted_gram[i, :] = s_predicted[i:i+n_gram]  # get data decoded by system
        s_gram[i, :] = s[i:i+n_gram]  # get origin data
        if s_predicted[i:i+n_gram] not in gram:
            gram[count, :] = s_predicted[i:i+n_gram]
            count += 1
        if s_gram[i:i+n_gram] not in gram:
            gram[count, :] = s[i:i+n_gram]
            count += 1

    gram2 = gram[0:count, :]

    min_zi = 0
    min_mu = 0
    for i in range(0, count):
        gram = gram2[i, :]
        s_predicted_count = 0
        s_count = 0
        for j in range(num_gram):
            if((gram == s_predicted_gram[j, :]).all()):
                s_predicted_count += 1
            if ((gram == s_gram[j, :]).all()):
                s_count += 1
        min_zi += min(s_predicted_count, s_count)
        min_mu += s_predicted_count
    return min_zi/min_mu
