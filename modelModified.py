"""
it includes some basic model and function of deepSC, but has been modified for mutual info joint training
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
    x_power = torch.sum(x ** 2)/ (batch_size * length * len_feature)
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
        self.frontDecoder = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.decoder = nn.TransformerEncoder(self.frontDecoder, num_layers=3)

        self.prediction = nn.Linear(128, 35632)
        self.softmax = nn.Softmax(dim=2)  # dim=2 means that it calculates softmax in the feature dimension

    def forward(self, inputs):
        embeddingVector = self.embedding(inputs)
        codeSent = self.encoder(embeddingVector)
        codeSent = self.denseEncoder1(codeSent)
        codeSent = self.denseEncoder2(codeSent)
        codeWithNoise = AWGN_channel(codeSent, 12)  # assuming snr = 12db
        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        codeReceived = self.decoder(codeReceived)
        infoPredicted = self.prediction(codeReceived)
        infoPredicted = self.softmax(infoPredicted)
        return infoPredicted, codeSent, codeWithNoise


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

def sample_batch(batch_size, sample_mode, x, y):  # used to sample data for mutual info system
    length = x.shape[0]
    if sample_mode == 'joint':
        index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[index, :]
        batch_y = y[index, :]
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(length), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[joint_index, :]
        batch_y = y[marginal_index, :]
    batch = torch.cat((batch_x, batch_y), 1)

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
            result -= torch.sum(label_term * torch.log(output_term + delta)) / length
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
