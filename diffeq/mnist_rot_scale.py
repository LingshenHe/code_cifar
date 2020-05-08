import torch
import torch.nn as nn
from scale_rot import SR_group
import numpy as np

class sr_base(nn.Module):
    def __init__(self, n1, flip, n2, r, in_channels=1, num_class=10, structure=[8,8,'max',16,16,'max',32,32], dropout=[0.1,0.2]):
        super(sr_base, self).__init__()
        self.g=SR_group(n1, flip, n2, r)
        type=('trivial', in_channels)
        main=[]

        for i in structure:
            if (isinstance(i, int)):
                main.append(self.g.conv(type, ('regular', i)))
                type=('regular', i)
                main.append(self.g.norm(type, momentum=0.1))
                main.append(nn.Dropout(dropout[0]))
                main.append(nn.ReLU())
            elif(i=='max'):
                main.append(self.g.MaxPool(type))
            elif(i=='avg'):
                main.append(self.g.AvgPool(type))
        self.main=nn.Sequential(*main)
        self.pool=self.g.GroupPool(type)
        type=('trivial', type[1])
        self.drop=nn.Dropout(dropout[1])
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(type[1], 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, num_class),
        )
        

    def forward(self, x):
        x=self.main(x)
        x=self.pool(x)
        x=x.reshape(x.size(0),-1)
        x=self.drop(x)
        x=self.fully_net(x)
        return x

def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Parameters of the net: {}M'.format(params/(10**6)))


class cnn(nn.Module):
    def __init__(self, list=[64, 64,'max', 64, 128, 128, 256, 256], in_channels=1, num_class=10):
        super(cnn, self).__init__()
        channels=in_channels
        main=[]
        for i in list:
            if(isinstance(i, int)):
                main.append(nn.Conv2d(channels, i, 3, padding=1, bias=False))
                channels=i
                main.append(nn.BatchNorm2d(channels))
                main.append(nn.ReLU())
            elif(i=='max'):
                main.append(nn.MaxPool2d(2,2))
            elif(i=='avg'):
                main.append(nn.AvgPool2d(2,2))
        self.main=nn.Sequential(*main)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(channels, num_class)
        self.drop=nn.Dropout(0.7)

    def forward(self, x):
        x=self.main(x)
        x=self.pool(x)
        x=x.reshape(x.size(0),-1)
        x=self.drop(x)
        x=self.fc(x)
        return x

def test():
    x=torch.randn(10,1,10,10)
    # net=sr_base(4,False, 4, 0.9)
    net=cnn()
    y=net(x)
    print(y.shape)
    compute_param(net)
test()

