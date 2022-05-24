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
print(cnn)
if __name__ == '__main__':
    cnn.load_state_dict(torch.load('model_weights_2.pth'))
    cnn.eval()

    mypath = './hw5_data/test/' 
    classname = [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]
    total = 0
    correct = 0
    for num, i in enumerate(classname):
        imgname = [ k for k in listdir(join(mypath, i)) if k.endswith('.jpg')] 
        print(i)
        for j in imgname:
            test_data = cv2.imread(join(mypath, i, j), cv2.IMREAD_GRAYSCALE)
            test_data = cv2.resize(test_data,(224,224))
            test_data = np.array(test_data)
            x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(test_data), dim=0),dim=0).type(torch.FloatTensor)
            x = Variable(x).cuda()
            predict = cnn(x).data.max(1, keepdim=True)[1].item()
            if predict == num:
                correct += 1
            total += 1
            print('imgname : ', j)
            print('predict class : ', predict)
    print('accuracy = ',correct/total)
