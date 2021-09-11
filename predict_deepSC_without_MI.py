"""
it's used to validate model trained from train.py
"""
import torch
import model
from model import calBLEU
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using " + str(device).upper())
model_path = './trainedModel/deepSC_without_MI.pth'


net = model.SemanticCommunicationSystem()
net.load_state_dict(torch.load(model_path, map_location = device))
net.to(device)
tokenizer = BertTokenizer.from_pretrained('bertModel')
bert_model = BertModel.from_pretrained('bertModel')

with open('data/corpus_10w.txt', 'r') as file:
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
    inputs = np.zeros((128, 30))  # store every id of corresponding word inside the sentence into the matrix
    num_list = []

    for i in range(128):
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

    for i in range(128):
        sen = text[i]
        sen_spilt = word_tokenize(sen)
        num = num_list[i]
        id_output = id_output_arr[i, :]  # get the id list of most possible word
        origin_sen = inputs[i, :]

        BLEU1 = calBLEU(1, id_output.cpu().detach.numpy(), origin_sen.cpu().detach().numpy(), num)
        BLEU2 = calBLEU(2, id_output.cpu().detach.numpy(), origin_sen.cpu().detach().numpy(), num)
        BLEU3 = calBLEU(3, id_output.cpu().detach.numpy(), origin_sen.cpu().detach().numpy(), num)
        BLEU4 = calBLEU(4, id_output.cpu().detach.numpy(), origin_sen.cpu().detach().numpy(), num)  # calculate BLEU
        BLEU_1_list.append(BLEU1)
        BLEU_2_list.append(BLEU2)
        BLEU_3_list.append(BLEU3)
        BLEU_4_list.append(BLEU4)

        sen_output = ''
        sen_input = ''
        id_output_np = id_output.cpu().detach().numpy
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

x = np.arange(0, 18, 3)
y1 = snr_BLEU_1_gram
y2 = snr_BLEU_2_gram
y3 = snr_BLEU_3_gram
y4 = snr_BLEU_4_gram
plt.title("deepSC without MI")
plt.xlabel("SNR")
plt.ylabel("BLEU")
plt.plot(x, y1, marker='D', label='1-gram')
plt.plot(x, y2, marker='D', label='2-gram')
plt.plot(x, y3, marker='D', label='3-gram')
plt.plot(x, y4, marker='D', label='4-gram')
plt.legend(loc='best')
plt.show()

print("All done!")

