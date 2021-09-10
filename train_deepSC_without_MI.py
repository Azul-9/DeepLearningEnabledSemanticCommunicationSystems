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
num_epoch = 10
save_path = './trainedModel/deepSC_without_MI.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(CorpusData(), batch_size= batch_size, shuffle= True)

model = SemanticCommunicationSystem()
model.to(device)
print('Using ' + str(device).upper())
optim = torch.optim.Adam(model.parameters(), lr=0.001)  # can change optimizer
lossFn = LossFn()

for epoch in range(num_epoch):
    train_bar = tqdm(dataloader)
    for i, data in enumerate(train_bar):
        [inputs, length_sen] = data  # get length of sentence without padding
        num_sample = inputs.size()[0]  # get how much sentence the system get
        inputs = torch.tensor(inputs[:, 0, :]).long()
        inputs = inputs.to(device)

        label = F.one_hot(inputs, num_classes = 35632).float()
        label = label.to(device)

        s_predicted = model(inputs)
        id_output = torch.argmax(s_predicted[0:4], dim=2)
        correct = (id_output == inputs[0:4])

        loss = lossFn(s_predicted, inputs, length_sen, num_sample, batch_size)
        loss.backward()
        optim.step()
        optim.zero_grad()

        print('batch: ', i, '  loss: ', loss.cpu().detach().numpy(),
              "  correct: ", torch.sum(correct).cpu().detach().numpy()/(batch_size*4))

torch.save(model.state_dict(), save_path)

