import torch
import torch.nn as nn
from utils import Group
import numpy as np
import time

from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

from PIL import Image

'''model'''

class rot_base(nn.Module):
    def __init__(self, n, flip, in_channels=1, num_class=10, structure=[24,28,'max',32,36,'max',64,64,'max'], dropout=[0.1,0]):
        super(rot_base, self).__init__()
        self.g=Group(n, flip)
        type=('trivial', in_channels)
        main=[]

        for i in structure:
            if (isinstance(i, int)):
                main.append(self.g.conv5x5(type, ('regular', i)))
                type=('regular', i)
                main.append(self.g.norm(type, momentum=0.1))
                main.append(nn.Dropout(dropout[0]))
                # main.append(nn.BatchNorm2d(self.g.rep[type[0]].dim*type[1]))
                main.append(nn.ReLU())
            elif(i=='max'):
                # main.append(nn.MaxPool2d(2,2))
                main.append(self.g.MaxPool(type))
            elif(i=='avg'):
                # main.append(nn.AvgPool2d(2,2))
                main.append(self.g.AvgPool(type))
        # main.append(self.g.norm(type, momentum=0.1))
        self.main=nn.Sequential(*main)
        # self.conv=self.g.conv5x5(type, ('trivial', type[1]))
        self.pool=self.g.GroupPool(type)
        type=('trivial', type[1])
        # self.bn=self.g.norm(('trivial', type[1]), momentum=0.1)
        # self.bn=nn.BatchNorm2d(self.g.rep[type[0]].dim*type[1])
        # self.pool=nn.AdaptiveAvgPool2d(1)
        # self.fc=nn.Linear(type[1],num_class)
        self.drop=nn.Dropout(dropout[1])
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(type[1], 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, num_class),
        )
        

    def forward(self, x):
        x=self.main(x)
        # x=self.bn(self.conv(x))
        x=self.pool(x)
        x=x.reshape(x.size(0),-1)
        x=self.drop(x)
        # x=self.fc(x)
        x=self.fully_net(x)
        return x

class cnn(nn.Module):
    def __init__(self, list=[64, 64,'max', 128, 128, 128, 256, 256], in_channels=1, num_class=10):
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


def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print('Parameters of the net: {}M'.format(params/(10**6)))
    return params

def test():
    x=torch.randn(10,1,10,10)
    net=rot_base(8,True)
    y=net(x)
    print(y.shape)
    compute_param(net)
    compute_param(net.conv)


'''dataset'''
bs=64
class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "../data/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "../data/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)
resize1 = Resize(84)
resize2 = Resize(28)

totensor = ToTensor()

train_transform = Compose([
    resize1,
    RandomRotation(180, resample=Image.BILINEAR, expand=False),
    resize2,
    totensor,
])

mnist_train = MnistRotDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=bs, shuffle=True)


test_transform = Compose([
    totensor,
])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=512)




'''train'''




learning_rate=0.5e-3
wd=2e-3
drop=[0., 0]
t1=time.time()
net=rot_base(16,False, dropout=drop)
t2=time.time()
print('init {}s'.format(t2-t1))
if(net.g.flip==True):
    group_type="D"+str(net.g.n)
else:
    group_type='C'+str(net.g.n)
# net=cnn()
param=compute_param(net)
model = nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
loss_function = torch.nn.CrossEntropyLoss()

# schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.8)
start=26
best=0.
for epoch in range(1,70):
    if(epoch==26):
        learning_rate=learning_rate*0.1
    elif(epoch==51):
        learning_rate=learning_rate*0.1
    t1=time.time()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    model.train()
    total = 0
    correct = 0
    print('The net is '+group_type)
    print('Parameters of the net: {}M'.format(param/(10**6)))
    print('lr: {}, wd: {}, dropout: {}, {},  batchsize: {}'.format(learning_rate, wd, drop[0], drop[1], bs))
    for i, (x, t) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.cuda()
        t = t.cuda()

        y = model(x)
        _, prediction = torch.max(y.data, 1)
        total += t.shape[0]
        correct += (prediction == t).sum().item()
        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()
    print(f"epoch {epoch} | train accuracy: {correct/total*100.}")
    if(epoch>=start):
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.cuda()
                t = t.cuda()
                
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")
        if(correct/total*100.>best):
            best=correct/total*100
            print('Best test acc: {}'.format(best))
    t2=time.time()
    print('Comsuming {}s'.format(t2-t1))
    
    print('\n')
    
print('Best test acc: {}'.format(best))


        



                

