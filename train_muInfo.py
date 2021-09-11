"""
it's used to train a mutual inforamtion system
"""

import model
import torch
from matplotlib import pyplot as plt


num_epoch = 200

save_path = './trainedModel/MutualInfoSystem.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using: " + str(device).upper())

net = model.MutualInfoSystem()
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.002)

muInfo = []
for i in range(num_epoch):
    batch_joint = torch.tensor(model.sample_batch(40, 'joint')).to(device)
    batch_marginal = torch.tensor(model.sample_batch(40, 'marginal')).to(device)
    t = net(batch_joint)
    et = torch.exp(net(batch_marginal))
    loss = -(torch.mean(t) - torch.log(torch.mean(et)))

    print('epoch: {}  '.format(i + 1))
    print(-loss.cpu().detach().numpy())
    muInfo.append(-loss.cpu().detach().numpy())

    loss.backward()
    optim.step()
    optim.zero_grad()

torch.save(net.state_dict(), save_path)
plt.title('train mutual info system')
plt.xlabel('Epoch')
plt.ylabel('Mutual Info')
plt.plot(muInfo)
plt.show()
print('All done!')

