"""
it's used to train a semantic communication system without mutual information model
"""
from torch.utils.data import DataLoader
from data_process import CorpusData
from model import SemanticCommunicationSystem
from model import LossFn
import torch
from tqdm import tqdm
import torch.nn.functional as F

batch_size = 128
num_epoch = 3
save_path = './trainedModel/deepSC_without_MI.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(CorpusData(), batch_size= batch_size, shuffle= True)

net = SemanticCommunicationSystem()
net.to(device)
print('Using ' + str(device).upper())
optim = torch.optim.Adam(net.parameters(), lr=0.001)  # can change optimizer
lossFn = LossFn()

for epoch in range(num_epoch):
    train_bar = tqdm(dataloader)
    for i, data in enumerate(train_bar):
        [inputs, length_sen] = data  # get length of sentence without padding
        num_sample = inputs.size()[0]  # get how much sentence the system get
        inputs = inputs[:, 0, :].clone().detach().requires_grad_(True).long()  # .long used to convert the tensor to long format
        # in order to fit one_hot function TODO
        inputs = inputs.to(device)

        label = F.one_hot(inputs, num_classes = 35632).float()
        label = label.to(device)

        s_predicted = net(inputs)

        loss = lossFn(s_predicted, label, length_sen, num_sample, batch_size)
        loss.backward()
        optim.step()
        optim.zero_grad()

        print('  loss: ', loss.cpu().detach().numpy())

torch.save(net.state_dict(), save_path)
print("All done!")

