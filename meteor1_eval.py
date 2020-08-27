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
import time
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
        x = x.view(-1, 16 * 3 * 3)
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

#-----------------------------------------------
# main
#-----------------------------------------------
def load_nn(model_path = './meteor1_weight.pth' ): #'model.pth'
    # 学習済データ読込    
    net = Net()
    #model_path = './meteor1_weight.pth' #'model.pth'
    net.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(net)

    return net

def nn_eval( net, img_fn ):
    epochs = 1
    _batch_size = 1 #4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #指定する画像フォルダ
    #train_path ='./tmp/data_20200501/train'
    test_path  ='./tmp/data/test'
    filename = test_path+'/meteor/1.png'
    if os.path.exists( filename ) :
        os.remove(filename)
    print(img_fn, filename)
    shutil.copy(img_fn, filename)

    # 取り込んだデータに施す処理を指定
    data_transform = transforms.Compose([
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # train data読み込み
    #meteor_dataset = datasets.ImageFolder(root=train_path,
    #                                       transform=data_transform)
    #dataset_loader = torch.utils.data.DataLoader(meteor_dataset,
    #                                         batch_size=_batch_size, shuffle=True, num_workers=4)

    # test data読み込み
    meteor_testset = datasets.ImageFolder(root=test_path,
                               transform=data_transform)
    dataset_testloader = torch.utils.data.DataLoader(meteor_testset, batch_size=_batch_size,
                                         shuffle=False, num_workers=4)

    classes = ('meteor', 'plane', 'other')



    #print(meteor_dataset[0])
    #print(meteor_testset[11])

    #my_testiter = iter(dataset_testloader)
    #images, labels = my_testiter.next()
    #imshow(torchvision.utils.make_grid(images))
    
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
                print("nn ans:",val_preds)
        
#   学習過程を記録
        #epoch_loss = running_loss/len(dataset_loader.dataset)
        #epoch_acc = running_corrects.float()/ len(dataset_loader.dataset)
        #running_loss_history.append(epoch_loss)
        #running_corrects_history.append(epoch_acc)
    
        #val_epoch_loss = val_running_loss/len(dataset_testloader.dataset)
#     print('len-detaset_testloader :'+str(len(detaset_testloader)))
#     print('len-detaset_testloader :'+str(len(detaset_testloader.dataset)))
        #val_epoch_acc = val_running_corrects.float()/len(dataset_testloader.dataset)
        #val_running_loss_history.append(val_epoch_loss)
        #val_running_corrects_history.append(val_epoch_acc)
    
        #print('epoch *', (e+1))
        #print('training   loss: {:.4f}, training acc   {:.4f}'.format(epoch_loss, epoch_acc.item()))
        #print('validation loss: {:.4f}, validation acc {:.4f}'.format(val_epoch_loss,val_epoch_acc.item()))
    
    #net.to('cpu')
    #torch.save(net.state_dict(), './meteor1_weight.pth')

    return ( val_preds[0].item() )

def rename_nn( img_fn, ans ):
    if not os.path.exists( img_fn ) :
        print('file not found.'+img_fn)
        return
    if not os.path.isfile( img_fn ) :
        print('file not found.'+img_fn)
        return
    st = '00s'+str(ans)+'.png'
    if ( img_fn.endswith('00s.png')) :
        st_org = '00s.png'
    elif ( img_fn.endswith('00s0.png')) :
        st_org = '00s0.png'
    elif ( img_fn.endswith('00s1.png')) :
        st_org = '00s1.png'
    elif ( img_fn.endswith('00s2.png')) :
        st_org = '00s2.png'
    fn_org = img_fn
    img_fn.replace( st_org, st)
    os.rename(fn_org, img_fn)

if __name__ == '__main__':
  
    #指定する画像フォルダ
    path_dir ='./tmp/data_20200427/'
    target_dir ='C:/temp/'
    img_fn ='./tmp/20200311_184214_276_00s2.png'
  
    t1 = time.time()
    #net = load_nn('./meteor1_weight_0505.pth' )
    net = load_nn( )
    t2 = time.time()
    ans = nn_eval( net, img_fn )
    t3 = time.time()
    elapsed_time1 = t2-t1
    elapsed_time2 = t3-t2
    print(img_fn+' : ' + str(ans))
    print(elapsed_time1,elapsed_time2)