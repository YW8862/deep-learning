import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.nn as nn
import torch.nn.functional as F

#准备数据集
def get_dataloader(train = True):
    dataset = MNIST("./data", train = train, download = True, transform = Compose([ToTensor(),
                                                                            Normalize((0.1307,),(0.3081,))
                                                                            ]))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    return dataloader


#构建网络模型
class Mnist(nn.Module):
    def __init__(self):
        super(Mnist,self).__init__()
        self.fc1 = nn.Linear(1*28*28,28)
        self.fc2 = nn.Linear(28,10)
    def forward(self,x):

        #形状的修改
        x = x.view([-1,1*28*28])
        #全链接
        x = self.fc1(x)
        #激活函数
        x = F.relu(x)
        #输出层
        x = self.fc2(x)

        x = F.log_softmax(x)
        return x

model = Mnist()
optimizer = Adam(model.parameters(),lr = 0.001)
if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))
def train(epoch):
    dataloader = get_dataloader()
    for idx,(input, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output,target)#得到损失
        loss.backward()
        optimizer.step()

        print(loss.item())
            # 模型保存
        if idx % 100 == 0:
            torch.save(model.state_dict(), './model/model.pkl')
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")


if __name__ == '__main__':
    for i in range(3):
        train(i)

