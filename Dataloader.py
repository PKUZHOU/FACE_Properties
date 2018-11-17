import torch
import  torch.utils.data as data
import os
import random
import cv2
import numpy as np

def transform(img):
    if random.random<0.5:
        img = cv2.flip(img)
    img = cv2.resize(img,(32,32))#.astype(np.float32)
    img = torch.Tensor(img)
    img = img/255.
    img = img.permute(2,0,1)
    return img

class PoseDateset(data.Dataset):
    def __init__(self,root = '/datasets/tmp/nvme1/POSE_data/',train = True,transform = None):
        self.root = root
        self.pos = []
        self.neg = []
        self.transform = transform
        self.train = train
        self.len_train = 420000
        self.len_test = 20000
        print("preparing data...")
        self.get_lists()
        self.train_pos = self.pos[:self.len_train]
        self.train_neg = self.neg[:self.len_train]
        self.test_pos = self.pos[self.len_train:self.len_train+self.len_test]
        self.test_neg = self.neg[self.len_train:self.len_train+self.len_test]
        print("data prepared!")
    def get_lists(self):

        pos_dir = self.root+'front_mid_mid_crop'
        files = os.listdir(pos_dir)
        pos_files = []
        for file in files:
            if file.split('.')[-1] == "jpg":
                pos_files.append(pos_dir+'/'+file)
        self.pos = pos_files
        for kind in os.listdir(self.root):
            if kind!='front_mid_mid_crop':
                # print kind
                for file in os.listdir(self.root+kind):
                    if file.split('.')[-1] == "jpg":
                        self.neg.append(self.root+kind+'/'+file)

    def __getitem__(self, index):
        if self.train:
            if(index%2==0):
                index = index/2
                filename = self.train_pos[index]
                label = 1
            elif(index%2==1):
                index = (index-1)/2
                filename = self.train_neg[index]
                label = 0
        if not self.train:
            # if (index % 2 == 0):
            #     index = index / 2
            #     filename = self.test_pos[index]
            #     label = 1
            # elif (index % 2 == 1):
            #     index = (index - 1) / 2
            #     filename = self.test_neg[index]
            #     label = 0
            filename = self.test_neg[index]
            label = 0
        img = cv2.imread(filename)
        if self.transform!=None:
            img = self.transform(img)
        else:
            img = torch.Tensor(img)
        label = torch.Tensor([label]).long()
        return img,label

    def __len__(self):
        if self.train:
            return self.len_train*2-1
        else:
            return self.len_test


class Multilable_Dateset(data.Dataset):
    def __init__(self,root = 'Mdata.txt',train = True,transform = None):
        self.root = root
        self.data = {}
        self.transform = transform
        self.train = train
        print("preparing data...")
        self.imgs = []
        self.lables = []
        self.get_lists()

        print("data prepared!")
    def get_lists(self):

        with open(self.root) as f:
            annos = f.readlines()
            for line in annos:
                line = line.strip().split(' ')
                file_path = line[0]
                nokouzhao,kouzhao= float(line[1]),float(line[2])
#                nokouzhao,kouzhao,nomojing,mojing = float(line[1]),float(line[2]),float(line[3]),float(line[4])
                self.data[file_path] = [nokouzhao,kouzhao]
        num = 0
        for key in self.data.keys():
            try:
                label = self.data[key]
                img = cv2.imread(key)
                if self.transform != None:
                    img = self.transform(img)                 
                label = torch.Tensor(label)
                self.lables.append(label)
                self.imgs.append(img)
                num+=1
                if (num%100==0):
                    print(num)
            except:
                print("broken")
                pass

    def __getitem__(self, index):
        # if self.train:
        # filename = self.data.keys()[index]
        # label = self.data[filename]
        label = self.lables[index]
        #label[0]/=45.
        #label[1]/=45.
        #label[2]/=45.
        img = self.imgs[index]
        # if self.transform!=None:
        #     img = self.transform(img)
        # else:
        #     img = torch.Tensor(img)
        # label = torch.Tensor(label)
        return img,label

    def __len__(self):
        return len(self.lables)



if __name__ == '__main__':
    Data = Multilable_Dateset(transform=transform)
    img,label = Data[0]
    print(label.data)
