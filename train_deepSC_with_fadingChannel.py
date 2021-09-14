"""
it's used to train a deepSC which includes a rayleigh fading channel
"""

import torch
import modelModifiedForFadingChannel
import numpy as np
import torch.nn.functional as F
from data_process import CorpusData
from torch.utils.data import DataLoader
from tqdm import tqdm


batch_size = 256
num_epoch = 2
save_path = './trainedModel/deepSC_with_fadingChannel.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using ' + str(device).upper())
dataloader = DataLoader(CorpusData(), batch_size= batch_size, shuffle= True)

P_hdB = np.array([0, -8, -17, -21, -25])  # Power characteristics of each channels(dB)
D_h = [0, 3, 5, 6, 8]  # Each channel delay(sampling point)
P_h = 10 ** (P_hdB / 10)  # Power characteristics of each channels
NH = len(P_hdB)  # Number of the multi channels
LH = D_h[-1] + 1  # Length of the channels(after delaying)
P_h = np.reshape(P_h, (len(D_h), 1))

def multipath_generator(num_sample):
    a = np.tile(np.sqrt(P_h / 2), num_sample)  # generate rayleigh stochastic variable
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

net = modelModifiedForFadingChannel.SemanticCommunicationSystem()
net = net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.001)
lossFn = modelModifiedForFadingChannel.LossFn()

for epoch in range(num_epoch):
    train_bar = tqdm(dataloader)
    for i, data in enumerate(train_bar):
        [inputs, length_sen] = data
        num_sample = inputs.size()[0]
        h_I, h_Q = multipath_generator(num_sample)
        inputs = inputs[:, 0, :].clone().detach().requires_grad_(True).long()
        inputs = inputs.to(device)

        label = F.one_hot(inputs, num_classes=35632).float()
        label = label.to(device)
        s_predicted = net(inputs, h_I, h_Q)

        loss = lossFn(s_predicted, label, length_sen, num_sample, batch_size)
        loss.backward()
        optim.step()
        optim.zero_grad()

        print('  loss: ', loss.cpu().detach().numpy())

torch.save(net.state_dict(), save_path)
print("All done!")