"""
it's used to train a model guided by deepSC and mutual information system
attention that it won't modify mutual info model, only deepSC's improvement will be stored
"""

import torch
from torch.utils.data import DataLoader
import modelModified
from tqdm import tqdm
from data_process import CorpusData
import torch.nn.functional as F

batch_size = 128
num_epoch = 3
lamda = 0.1  # it's used to control how much the muInfo will affect deepSC model
save_path = './trainedModel/deepSC_with_MI.pth'
deepSC_path = 'trainedModel/deepSC_without_MI.pth'
muInfo_path = 'trainedModel/MutualInfoSystem.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using ' + str(device).upper())

dataloader = DataLoader(CorpusData(), batch_size= batch_size, shuffle=True)
scNet = modelModified.SemanticCommunicationSystem()
scNet.load_state_dict(torch.load(deepSC_path, map_location=device))
scNet.to(device)
muInfoNet = modelModified.MutualInfoSystem()
muInfoNet.load_state_dict(torch.load(muInfo_path, map_location=device))
muInfoNet.to(device)

optim = torch.optim.Adam(scNet.parameters(), lr=0.001)
lossFn = modelModified.LossFn()

for epoch in range(num_epoch):
    train_bar = tqdm(dataloader)
    for i, data in enumerate(train_bar):
        [inputs, length_sen] = data  # get length of sentence without padding
        num_sample = inputs.size()[0]  # get how much sentence the system get
        inputs = inputs[:, 0, :].clone().detach().requires_grad_(True).long()  # .long used to convert the tensor to long format
        # in order to fit one_hot function
        inputs = inputs.to(device)

        label = F.one_hot(inputs, num_classes=35632).float()
        label = label.to(device)

        [s_predicted, codeSent, codeWithNoise] = scNet(inputs)

        x = torch.reshape(codeSent, (-1, 16))  # get intermediate variables to train mutual info sys
        y = torch.reshape(codeWithNoise, (-1, 16))

        batch_joint = modelModified.sample_batch(5, 'joint', x, y).to(device)
        batch_marginal = modelModified.sample_batch(5, 'marginal', x, y).to(device)

        t = muInfoNet(batch_joint)
        et = torch.exp(muInfoNet(batch_marginal))
        MI_loss = torch.mean(t) - torch.log(torch.mean(et))
        SC_loss = lossFn(s_predicted, label, length_sen, num_sample)

        loss = SC_loss + torch.exp(-MI_loss) * lamda
        loss.backward()
        optim.step()
        optim.zero_grad()

        print("Total Loss: {}, Mutual Loss: {}, SC Loss: {}".format(loss.cpu().detach.numpy(),
                                                                    MI_loss.cpu().detach.numpy(),
                                                                    SC_loss.cpu().detach.numpy()))

torch.save(scNet.state_dict(), save_path)
print("All done!")