import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

import torch.nn.init as init

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

class PoseNet(nn.Module):
    def __init__(self,num_lable = 2):
        super(PoseNet, self).__init__()
        self.conv1 = nn.Conv2d(3,28,kernel_size=3)
        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.conv2 = nn.Conv2d(28,48,3)
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode= True)
        self.fc1 = nn.Linear(2352,256)##(2352,256) ##768,256
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,num_lable)
        self.apply(weights_init)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class PoseLoss(nn.Module):
    def __init__(self,num_out = 4):
        super(PoseLoss,self).__init__()
        self.num_out = num_out
    def forward(self, pred,label):
        batch = pred.shape[0]
        loss = (pred-label)**2
        loss = loss.view(-1).sum()/batch
        return loss

if __name__ == '__main__':
    x = torch.ones(1,3,32,32)
    l = torch.ones(1,3)
    x = Variable(x)
    l = Variable(l)
    loss = PoseLoss()
    net = PoseNet(2)
    L = loss(net(x),l)
    print(L)
