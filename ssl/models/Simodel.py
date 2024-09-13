import torch
import numpy as np
import os
import random
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader


class Simodel(nn.Module):
    def __init__(self):
        super(Simodel,self).__init__()
        self.cnn11 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.bn11=nn.BatchNorm2d(32)
        self.cnn12 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.bn12=nn.BatchNorm2d(32)
        self.d1 = nn.Dropout2d(0.5)
        self.cnn21 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.bn21=nn.BatchNorm2d(64)
        self.cnn22 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.bn22=nn.BatchNorm2d(64)
        self.d2 = nn.Dropout2d(0.5)
        self.cnn31 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.bn31=nn.BatchNorm2d(128)
        self.cnn32 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.bn32=nn.BatchNorm2d(128)
        self.d3 = nn.Dropout2d(0.5)
        self.nn1= nn.Linear(128*16, 10)


    def forward(self,x):
        x= self.bn11(F.relu(self.cnn11(x)))
        x= self.bn12(F.relu(self.cnn12(x)))
        x=self.d1(F.max_pool2d(x,2,2))
        x= self.bn21(F.relu(self.cnn21(x)))
        x= self.bn22(F.relu(self.cnn22(x)))
        x=self.d2(F.max_pool2d(x,2,2))
        x= self.bn31(F.relu(self.cnn31(x)))
        x= self.bn32(F.relu(self.cnn32(x)))
        x=self.d3(F.max_pool2d(x,2,2))
        x=torch.flatten(x,1)
        return self.nn1(x)


class Simodel2(nn.Module):
    def __init__(self):
        super(Simodel2,self).__init__()
        self.cnn11 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.bn11=nn.BatchNorm2d(32)
        self.cnn12 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.bn12=nn.BatchNorm2d(32)
        self.d1 = nn.Dropout2d(0.5)
        self.cnn21 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.bn21=nn.BatchNorm2d(64)
        self.cnn22 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.bn22=nn.BatchNorm2d(64)
        self.d2 = nn.Dropout2d(0.5)
        self.cnn31 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.bn31=nn.BatchNorm2d(128)
        self.cnn32 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.bn32=nn.BatchNorm2d(128)
        self.d3 = nn.Dropout2d(0.5)
        self.nn1= nn.Linear(128*16, 10)


    def forward(self,x):
        x= self.bn11(F.relu(self.cnn11(x)))
        x= self.bn12(F.relu(self.cnn12(x)))
        x1=x
        x=self.d1(F.max_pool2d(x,2,2))
        x= self.bn21(F.relu(self.cnn21(x)))
        x= self.bn22(F.relu(self.cnn22(x)))
        x2=x
        x=self.d2(F.max_pool2d(x,2,2))
        x= self.bn31(F.relu(self.cnn31(x)))
        x= self.bn32(F.relu(self.cnn32(x)))
        x3=x
        x=self.d3(F.max_pool2d(x,2,2))
        x=torch.flatten(x,1)
        return self.nn1(x),[x1,x2,x3]




















def train(log_interval, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    batch_size=64
    test_batch_size=1000
    epochs=90
    lr=3e-4
    weight_decay=0
    #gamma=0.
    seed=1
    log_interval=150
    save_model=True
    
    #use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    #kwargs = {'num_workers': 1, 'pin_memory': True}
    transform = T.Compose([T.ToTensor()])
    test_data = dset.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = DataLoader(test_data, batch_size=test_batch_size,num_workers=2,shuffle=True, pin_memory=True)
    transform = T.Compose([T.ToTensor()])
    train_data=dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader=DataLoader(train_data, batch_size=batch_size,num_workers=2,shuffle=True, pin_memory=True)

    model = Simodel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(epochs):
        train(log_interval,model,trainloader,optimizer,epoch)
        test(model,testloader)

    if save_model:
        torch.save(model.state_dict(), "cifar_cnn_32.pt")
    

if __name__=='__main__': 
    main()
    




        








