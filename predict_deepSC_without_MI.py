"""
it's used to validate model trained from train.py
attention it's wired that argument snr failed to be transported into model and fading channel cannot update it's argument
so i put them into the same file instead of importing from outside
"""
import torch
from model import calBLEU
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using " + str(device).upper())
model_path = './trainedModel/deepSC_without_MI.pth'


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
        codeWithNoise = AWGN_channel(codeSent, snr)  # assuming snr = 12db
        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        codeSemantic = self.decoder(codeReceived, code)
        codeSemantic = self.prediction(codeSemantic)
        info = self.softmax(codeSemantic)
        return info


net = SemanticCommunicationSystem()
net.load_state_dict(torch.load(model_path, map_location = device))
net.to(device)
tokenizer = BertTokenizer.from_pretrained('bertModel')
bert_model = BertModel.from_pretrained('bertModel')

with open('data/corpus_10w.txt', 'r', encoding='utf-8') as file:
    start = ""
    end = ""
    text = [start + line.strip() + end for line in file]
with open('data/id_dic_10w.pkl', 'rb') as file:
    id_dic = pickle.load(file)
with open('data/word_dic_10w.pkl', 'rb') as file:
    word_dic = pickle.load(file)

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

    inputs = torch.tensor(inputs).long()
    inputs = inputs.to(device)
    label = F.one_hot(inputs, num_classes = 35632).float()  # convert to tensor
    label = label.to(device)

    s_predicted = net(inputs)
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
plt.suptitle("deepSC without MI")
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

