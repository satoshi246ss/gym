# -*- coding: utf-8 -*-

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
#import Image

#Model Definition
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3CH 24x24
        self.pool = nn.MaxPool2d(2, 2)   # (24-4)/2 = 10  10x10
        self.conv2 = nn.Conv2d(6, 16, 5) # 6ch->16ch (10-4)=6 6x6 -> 3x3
        self.fc1 = nn.Linear(16 * 3 * 3, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape,x.size(0)) #torch.Size([4, 16, 3, 3])
        #x = x.view(-1, 16 * 6 * 6)
        x = x.view(-1, 16 * 3 * 3) #4
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# imshow関数は次に説明するけどちょっと先走って使う．
def imshow(img):
    img = img #/ 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    #指定する画像フォルダ
    train_path ='./tmp/data_20200501/train'
    test_path  ='./tmp/data_20200501/test'

    epochs = 500
    _batch_size = 1 #4
    # 取り込んだデータに施す処理を指定
    data_transform = transforms.Compose([
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # train data読み込み
    meteor_dataset = datasets.ImageFolder(root=train_path,
                                           transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(meteor_dataset,
                                             batch_size=_batch_size, shuffle=True,
                                             num_workers=4)

    # test data読み込み
    meteor_testset = datasets.ImageFolder(root=test_path,
                               transform=data_transform)
    dataset_testloader = torch.utils.data.DataLoader(meteor_testset, batch_size=_batch_size,
                                         shuffle=False, num_workers=4)

    classes = ('meteor', 'other')

    for i, data in enumerate(dataset_loader, 0):
        inputs, labels = data
        print (inputs.size())
        print (labels.size())
    
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    print(net)
    #print(meteor_dataset[0])
    #print(meteor_testset[11])

    my_testiter = iter(dataset_testloader)
    images, labels = my_testiter.next()
    imshow(torchvision.utils.make_grid(images))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):
  
        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0
  
        for inputs, labels in dataset_loader:
        #     DataLoaderのバッチサイズごとにforで取り出して計算
        #     ここのforの処理が終わると1エポック
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
    
            #     一旦パラメーターの勾配をゼロにして
            optimizer.zero_grad()
            #     勾配の計算
            loss.backward()
            #     学習
            optimizer.step()
    
#     分類わけなので、もっとも数字が大きいものをpredictとする
#     バッチ処理しているので2次元目で比較
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
#     ラベルと合っているものを足し合わせてaccuracy計算
            running_corrects += torch.sum(preds == labels.data)
  
        else:
#     pytorchでは勾配の計算の高速化のため、パラメーターを保持しているがテスト時はいらないので止める
            with torch.no_grad():
                for val_inputs, val_labels in dataset_testloader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = net(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)
        
                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(val_preds == val_labels.data)
        
#   学習過程を記録
            epoch_loss = running_loss/len(dataset_loader.dataset)
            epoch_acc = running_corrects.float()/ len(dataset_loader.dataset)
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc)
    
            val_epoch_loss = val_running_loss/len(dataset_testloader.dataset)
#     print('len-detaset_testloader :'+str(len(detaset_testloader)))
#     print('len-detaset_testloader :'+str(len(detaset_testloader.dataset)))
            val_epoch_acc = val_running_corrects.float()/len(dataset_testloader.dataset)
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)
    
            print('epoch *', (e+1))
            print('training   loss: {:.4f}, training acc   {:.4f}'.format(epoch_loss, epoch_acc.item()))
            print('validation loss: {:.4f}, validation acc {:.4f}'.format(val_epoch_loss,val_epoch_acc.item()))
    
    net.to('cpu')
    torch.save(net.state_dict(), './meteor1_weight_0505.pth')

if __name__ == '__main__':
  
    #指定する画像フォルダ
    path_dir ='./tmp/data_20200427/'
    target_dir ='C:/temp/'
    main()