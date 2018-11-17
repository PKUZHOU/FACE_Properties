#/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import DataLoader
from Dataloader import Multilable_Dateset,transform
from network import PoseNet,PoseLoss

batch = 256
epoch = 30

trainset =Multilable_Dateset(train=True,transform=transform)
trainloader = DataLoader(trainset,batch_size=batch,shuffle=True,num_workers=8)


def adjust_learning_rate(optimizer,decay_rate = 0.1): #学习率每次乘0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def train():

    net = PoseNet(2) #out put is 3 dimentions
    net.train(True)
    net.cuda() #使用显卡训练
    opt = torch.optim.SGD(net.parameters(),lr = 0.001,momentum=0.9,weight_decay=5e-4) #SGD 参数
    criterion = PoseLoss()

    for epc in range(epoch):
        avgloss = torch.Tensor([0.])
        batchs = 0
        if epc in [3,10,15,20]: # 学习率降低的epc
            adjust_learning_rate(opt)
        torch.save(net.state_dict(),'M_models/model_'+str(epc)+'.pkl') #保存模型
        for index,data in enumerate(trainloader):
            image,label = data
            image = image.cuda()
            label = label.cuda()
            output = net(image)
            loss = criterion(output,label)
            avgloss+=loss.cpu().data
            batchs+=1
            loss.backward()
            opt.step()
            opt.zero_grad()

            if(index%10==0):
                print("epoch : {}  batch : {}  loss : {:.3f} avg_loss :{:.3f} lr: {} ".format(epc,index,float(loss.data[0]),float(avgloss.data[0])/batchs,opt.param_groups[0]['lr']))

if __name__ == '__main__':
    train()
