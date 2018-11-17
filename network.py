import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
import torch.nn as nn
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

class Property_Net(nn.Module):
    def __init__(self):
        super(Property_Net, self).__init__()
        self.conv1 = nn.Conv2d(3,28,kernel_size=3)
        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.conv2 = nn.Conv2d(28,48,3)
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode= True)
        self.fc1 = nn.Linear(2352,256)##(2352,256) ##768,256
        self.fc2 = nn.Linear(256,128)
        self.glass_pred = nn.Linear(128,2)
        self.mask_pred = nn.Linear(128,2)
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
        glass_pred = self.glass_pred(x)
        mask_pred = self.mask_pred(x)
        return [glass_pred,mask_pred]

class PropertyLoss(nn.Module):
    def __init__(self):
        super(PropertyLoss, self).__init__()
        self.glass_criterian = nn.CrossEntropyLoss(size_average=True)
        self.mask_criterian = nn.CrossEntropyLoss(size_average=True)
    def forward(self, pred, label):
        """
        pred: [glass_pred, mask_pred]
        label: [glass_label,  mask_label]
        """
        glass_pred = pred[0]
        mask_pred = pred[1]

        glass_label = label[:,0].view(-1)
        mask_label = label[:,1].view(-1)

        glass_loss = self.glass_criterian(glass_pred,glass_label)
        mask_loss = self.mask_criterian(mask_pred,mask_label)
        loss = glass_loss+mask_loss
        return loss

if __name__ == '__main__':
    x = Variable(torch.ones(100,3,32,32))
    l = [Variable(torch.ones(100).long()),Variable(torch.ones(100).long())]
    # x = Variable(x)
    # l = Variable(l)
    loss = PropertyLoss()
    net = Property_Net()
    L = loss(net(x),l)
    print(L)
