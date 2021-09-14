"""
it's used to validate model trained from fading channel
attention it's wired that argument snr failed to be transported into model and fading channel cannot update it's argument
so i put them into the same file instead of importing from outside
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from matplotlib import pyplot as plt
from modelModifiedForFadingChannel import calBLEU


def embedding(input_size, output_size):  # embedding layer, the former is the size of dic and
    # the latter is the dimension of the embedding vector
    return nn.Embedding(input_size, output_size)

def dense(input_size, output_size):  # dense layer is a full connection layer and used to gather information
    return torch.nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU()
    )

def fading_channel(x, h_I, h_Q, snr):
    [batch_size, length, feature_length] = x.shape
    x = torch.reshape(x, (batch_size, -1, 2))
    x_com = torch.complex(x[:, :, 0], x[:, :, 1])
    x_fft = torch.fft.fft(x_com)
    h = torch.complex(torch.tensor(h_I), torch.tensor(h_Q))
    h_fft = torch.fft.fft(h, feature_length * length//2).to(device)
    y_fft = h_fft * x_fft
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(y_fft ** 2) / (length * feature_length * batch_size // 2)
    npower = xpower / snr
    n = torch.randn(batch_size, feature_length * length // 2, device=device) * npower
    y_add = y_fft + n
    y_add = y_add / h_fft
    y = torch.fft.ifft(y_add)
    y_tensor = torch.zeros((y.shape[0], y.shape[1], 2), device=device)
    y_tensor[:, :, 0] = y.real
    y_tensor[:, :, 1] = y.imag
    y_tensor = torch.reshape(y_tensor, (batch_size, length, feature_length))

    return y_tensor

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

    def forward(self, inputs, h_I, h_Q):
        embeddingVector = self.embedding(inputs)
        code = self.encoder(embeddingVector)
        codeSent = self.denseEncoder1(code)
        codeSent = self.denseEncoder2(codeSent)
        codeWithNoise = fading_channel(codeSent, h_I, h_Q, snr)
        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        codeReceived = self.decoder(codeReceived, code)
        infoPredicted = self.prediction(codeReceived)
        infoPredicted = self.softmax(infoPredicted)
        return infoPredicted


num_epoch = 1
model_path = './trainedModel/deepSC_with_fadingChannel.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using ' + str(device).upper())

P_hdB = np.array([0, -8, -17, -21, -25])  # Power characteristics of each channels(dB)
D_h = [0, 3, 5, 6, 8]  # Each channel delay(sampling point)
P_h = 10 ** (P_hdB / 10)  # Power characteristics of each channels(dB)
NH = len(P_hdB)  # Number of the multi channels
LH = D_h[-1] + 1  # Length of the channels(after delaying)
P_h = np.reshape(P_h, (len(D_h), 1))

def multipath_generator(num_sample):
    a = np.tile(np.sqrt(P_h / 2), num_sample)
    A_h_I = np.random.rand(NH, num_sample) * a
    A_h_Q = np.random.rand(NH, num_sample) * a
    h_I = np.zeros((num_sample, LH))
    h_Q = np.zeros((num_sample, LH))

    i = 0
    for index in D_h:
        h_I[:, index] = A_h_I[i, :]
        h_Q[:, index] = A_h_Q[i, :]
        i += 1

    return h_I, h_Q

net = SemanticCommunicationSystem()
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)

with open("data/corpus_10w_test.txt", "r", encoding='utf-8') as f:
    start = ""
    end = ""
    text = [start + line.strip() + end for line in f]
with open('data/id_dic_10w.pkl', 'rb') as f:
    id_dic = pickle.load(f)
with open('data/word_dic_10w.pkl', 'rb') as f:
    word_dic = pickle.load(f)

tokenizer = BertTokenizer.from_pretrained('bertmodel')
bert_model = BertModel.from_pretrained('bertmodel')

snr_BLEU_1_gram = []
snr_BLEU_2_gram = []
snr_BLEU_3_gram = []
snr_BLEU_4_gram = []
snr_sen_similarity_gram = []

for snr in range(1, 18, 3):
    BLEU_1_list = []
    BLEU_2_list = []
    BLEU_3_list = []
    BLEU_4_list = []
    sen_similarity_list = []
    inputs = np.zeros((256, 30))  # store every id of corresponding word inside the sentence into the matrix
    num_list = []

    for i in range(256):
        sen = text[i]  # get a sentence
        sen_spilt = word_tokenize(sen)  # get a list consist of words inside the sentence
        inputs_one_sen = np.zeros((1, 30))  # create a matrix to store the word split above
        num = 0
        for word in sen_spilt:
            inputs_one_sen[0, num] = id_dic[word]  # store the corresponding id of word into the matrix
            num += 1
            if num >= 30:
                break
        inputs[i, :] = inputs_one_sen
        num_list.append(num)  # used to store evert length of sentence

    h_I, h_Q = multipath_generator(256)
    inputs = torch.tensor(inputs).long()
    inputs = inputs.to(device)
    label = F.one_hot(inputs, num_classes=35632).float()  # convert to tensor
    label = label.to(device)

    s_predicted = net(inputs, h_I, h_Q)
    id_output_arr = torch.argmax(s_predicted, dim=2)

    for i in range(256):
        sen = text[i]
        sen_spilt = word_tokenize(sen)
        num = num_list[i]
        id_output = id_output_arr[i, :]  # get the id list of most possible word
        origin_sen = inputs[i, :]

        BLEU1 = calBLEU(1, id_output.cpu().detach().numpy(), origin_sen.cpu().detach().numpy(), num)
        BLEU2 = calBLEU(2, id_output.cpu().detach().numpy(), origin_sen.cpu().detach().numpy(), num)
        BLEU3 = calBLEU(3, id_output.cpu().detach().numpy(), origin_sen.cpu().detach().numpy(), num)
        BLEU4 = calBLEU(4, id_output.cpu().detach().numpy(), origin_sen.cpu().detach().numpy(), num)  # calculate BLEU
        BLEU_1_list.append(BLEU1)
        BLEU_2_list.append(BLEU2)
        BLEU_3_list.append(BLEU3)
        BLEU_4_list.append(BLEU4)

        sen_output = ''
        sen_input = ''
        id_output_np = id_output.cpu().detach().numpy()
        for index in range(num):
            key = id_output_np[index]  # get the id of the word which go through the model
            sen_output += word_dic[key]  # convert id to the word
            sen_output += " "
            sen_input += sen_spilt[index]  # get the id of the original word
            sen_input += " "

        encoded_input = tokenizer(sen_input, return_tensors='pt')  # encode sentence to fit bert model
        bert_input = bert_model(**encoded_input).pooler_output  # get semantic meaning of the sentence
        encoded_input = tokenizer(sen_output, return_tensors='pt')
        bert_output = bert_model(**encoded_input).pooler_output
        sen_similarity = torch.sum(bert_input * bert_output) / (torch.sqrt(torch.sum(bert_input * bert_input))
                                                                * torch.sqrt(torch.sum(bert_output * bert_output)))
        sen_similarity_list.append(sen_similarity.cpu().detach().numpy())

    snr_BLEU_1_gram.append(np.mean(BLEU_1_list))
    snr_BLEU_2_gram.append(np.mean(BLEU_2_list))
    snr_BLEU_3_gram.append(np.mean(BLEU_3_list))
    snr_BLEU_4_gram.append(np.mean(BLEU_4_list))  # get mean value after processing 128 sentences
    snr_sen_similarity_gram.append(np.mean(sen_similarity_list))

    print("SNR: {} has finished".format(snr))

x = np.arange(1, 18, 3)
y1 = snr_BLEU_1_gram
y2 = snr_BLEU_2_gram
y3 = snr_BLEU_3_gram
y4 = snr_BLEU_4_gram
y5 = snr_sen_similarity_gram
plt.figure(figsize=(6.4, 9.6))
plt.suptitle("deepSC with Fading Channel")
plt.subplot(2, 1, 1)
plt.xlabel("SNR")
plt.ylabel("BLEU")
plt.plot(x, y1, marker='D', label='1-gram')
plt.plot(x, y2, marker='D', label='2-gram')
plt.plot(x, y3, marker='D', label='3-gram')
plt.plot(x, y4, marker='D', label='4-gram')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.xlabel("SNR")
plt.ylabel("Sentence Similarity")
plt.plot(x, y5, marker='D')
plt.show()

print("All done!")