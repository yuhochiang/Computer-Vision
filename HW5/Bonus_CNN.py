import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy import moveaxis
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from os import listdir 
from os.path import isfile, isdir, join
import csv

BATCH_SIZE = 10
LR = 0.001
num_epochs = 100
"""input(put in training_data file)"""
mypath='./hw5_data/train/' 
classname = [ f for f in listdir(mypath) if isdir(join(mypath,f)) ] 
inputarr = []
labelarr = []
#print('classname = ', classname)
for num, i in enumerate(classname):
    #print(num, i)
    imgname = [ k for k in listdir(join(mypath, i)) if k.endswith('.jpg')] 
    #print('imgname ', imgname)
    for j in imgname:
        train_data = cv2.imread(join(mypath, i, j), cv2.IMREAD_GRAYSCALE)
        train_data = cv2.resize(train_data,(224,224))
        inputarr.append(train_data)
        labelarr.append(num)

imgs = np.array(inputarr)
print(imgs.shape)
print(imgs.shape)
x = torch.unsqueeze(torch.from_numpy(imgs), dim=1).type(torch.FloatTensor)
y = torch.LongTensor(labelarr)


x, y = Variable(x).cuda(), Variable(y).cuda()

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

N_CLASS = 15
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        self.out = nn.Linear(25088, N_CLASS)

    def forward(self, x):
        conv_res = self.conv_layer(x)
        out = conv_res.view(conv_res.size(0), -1)
        output = self.out(out)
        return output

cnn = CNN().cuda()
# something about plotting
#plt.ion() 
#plt.show()
# something about plotting

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()  #Loss_func

def train():
    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            #forward
            prediction = cnn(batch_x)
            #print(prediction, '\nbatch : ', batch_y)
            loss = loss_func(prediction, batch_y)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 1 == 0:
                print('epoch[{}/{}], loss: {}'.format(epoch+1, num_epochs, loss.data))
    torch.save(cnn.state_dict(), 'model_weights_3.pth')
if __name__ == '__main__':
    train()
