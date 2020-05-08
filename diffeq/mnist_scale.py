import torch
import torch.nn as nn
from scale import scale_group
import numpy as np

class scale_base(nn.Module):
    def __init__(self, n, r, in_channels=1, num_class=10, structure=[12,12,'max',24,24,'max',48,48], dropout=[0.1,0.2]):
        super(scale_base, self).__init__()
        self.g=scale_group(n, r)
        type=('trivial', in_channels)
        main=[]

        for i in structure:
            if (isinstance(i, int)):
                main.append(self.g.conv5x5(type, ('regular', i)))
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

def test():
    x=torch.randn(10,1,10,10)
    net=scale_base(5,0.9)
    y=net(x)
    print(y.shape)
    compute_param(net)
test()