from utils import Group
import torch
import torch.nn as nn
import numpy as np
import math
from anti import Downsample



def kaiming_init(base, num_in, num_out, normal=True):
    '''
        base: the base of the conv_base or conv_fast
        num_in: number of representation in of the input in conv_base of conv_fast
        num_out: number of representation in of the output in conv_base of conv_fast
        normal: using the normal or constant initialization
    '''
    f=torch.sum(base*base)*num_in/(base.size(1))
    if(normal==True):
        weight=torch.sqrt(1/f)*torch.randn(num_in, num_out, base.size(0))
    else:
        weight=torch.sqrt(12/f)*(torch.rand(num_in, num_out, base.size(0))-0.5)
    return weight




class GroupBatchNorm(nn.Module):
    
    
    def __init__(self, num_rep, dim_rep, affine=False, momentum=0.1, track_running_stats=True):
        super(GroupBatchNorm,self).__init__()
        self.momentum=momentum
        self.bn=nn.BatchNorm3d(num_rep, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.num_rep=num_rep
        self.dim_rep=dim_rep
    
    
    def forward(self, x):
        shape=x.shape
        x=self.bn(x.reshape(x.size(0), self.num_rep, self.dim_rep, x.size(2),x.size(3)))
        x=x.reshape(shape)
        return x


class GroupPooling(nn.Module):
    def __init__(self, dim_rep, num_rep):
        super(GroupPooling, self).__init__()
        self.pool=nn.AdaptiveAvgPool3d(1)
        self.dim=dim_rep
        self.num_rep=num_rep

    def forward(self, x):
        size=x.size()
        x=x.reshape(x.size(0), self.num_rep, self.dim, x.size(2), x.size(3))
        x=self.pool(x)
        return x.reshape(x.size(0), x.size(1),1,1)



class conv_base(nn.Module):
    def __init__(self, base, num_in, num_out, stride, bias):
        super(conv_base, self).__init__()
        self.base=torch.nn.Parameter(base, requires_grad= False)
        self.bases=self.base.size(0)
        self.size=self.base.size(4)
        self.stride=stride
        self.param=torch.nn.Parameter(kaiming_init(base, num_in, num_out))
        self.dim_rep_in=self.base.size(2)
        self.dim_rep_out=self.base.size(1)
        self.num_in=num_in
        self.num_out=num_out
        self.bias=None
        if(bias==True):
            self.bias=nn.Parameter(torch.zeros(self.num_out), requires_grad=True)
            self.a=nn.Parameter(torch.ones(self.dim_rep_in), requires_grad=False)


    def forward(self,x):
            #get the kernel of the conv from the base
            bias=None
            kernel=torch.einsum('ijk,kmnpq->jminpq',self.param, self.base)\
                .reshape(self.num_out*self.dim_rep_out,self.num_in*self.dim_rep_in,self.size,self.size)
            if self.bias is not None:
                bias=torch.einsum('i,j->ij', self.bias, self.a).reshape(-1)
            return nn.functional.conv2d(x,kernel,bias=bias, stride=self.stride, padding=math.floor(self.size/2))




class SR_group:
    def __init__(self, n1, flip, n2, t):
        self.rot_group=Group(n1, flip)
        self.base3={('regular','regular'):None, ('regular','trivial'):None, ('trivial','trivial'):None,('trivial','regular'):None}
        self.base5={('regular','regular'):None, ('regular','trivial'):None, ('trivial','trivial'):None,('trivial','regular'):None}
        self.base1={('regular','regular'):None, ('regular','trivial'):None, ('trivial','trivial'):None,('trivial','regular'):None}
        self.n1=n1
        self.flip=flip
        self.n2=n2
        self.t=t
        self.dim=self.rot_group.dim*self.n2
        self.rep={'trivial':1, 'regular':self.dim}

    def base5x5(self, in_rep, out_rep):
        '''
            in: string indicate the representation type of input
            out: string indicate the representtation type of output
            ent the contant of scale transform. The group element will be of the form t^(-1), 1, t, t^2
        '''
        if(self.base5[(in_rep,out_rep)] is None):
            if(in_rep=='trivial' and out_rep=='regular'):
                base=[]
                s=torch.arange(self.n2).to(torch.float)
                for i in range(3, 8):
                    bas=torch.einsum('kijmn,t->ktijmn',self.rot_group.base(i, in_rep, out_rep), self.t**((i%5)*s))
                    base.append(bas.reshape(bas.size(0),bas.size(1)*bas.size(2), bas.size(3), bas.size(4), bas.size(5)))
                base=torch.cat(base, dim=0)
                shape=base.shape
                base,_=torch.qr(base.reshape(shape[0],-1).transpose(0,1))
                base=base.transpose(0,1).reshape(shape)
                self.base5[(in_rep,out_rep)]=base
            elif(in_rep=='regular' and out_rep=='regular'):
                base=[]
                n=self.n2*2-1
                s=torch.zeros(self.n2, self.n2, n)
                for i in range(self.n2):
                    s[i,::,(self.n2-1-i):n-i]=torch.eye(self.n2)
                s1=torch.arange(self.n2).to(torch.float)
                for i in range(3,8):
                    bas=torch.torch.einsum('kijmn, t, tpq->qktipjmn',self.rot_group.base(i, in_rep, out_rep), self.t**((i%5)*s1), s)
                    a1,a2,a3,a4,a5,a6,a7,a8=bas.size()
                    base.append(bas.reshape(a1*a2, a3*a4, a5*a6, a7, a8))
                    # print(i%5)
                    # print(torch.sum(base[-1][40,0:12]**2))
                    # print(base[-1][40,8])
                    # print(base[-1][40,12])
                    # print(base[-1][0,7])
                base=torch.cat(base, dim=0)
                shape=base.shape
                base,_=torch.qr(base.reshape(shape[0],-1).transpose(0,1))
                base=base.transpose(0,1).reshape(shape)
                self.base5[(in_rep,out_rep)]=base
            elif(in_rep=='regular' and out_rep=='trivial'):
                b=torch.einsum('pm,qn,ij->pqijmn', torch.eye(5), torch.eye(5), torch.ones(1, self.dim))
                a1,a2,a3,a4,a5,a6=b.shape
                b=b.reshape(a1*a2,a3,a4,a5,a6)
                shape=b.shape
                b,_=torch.qr(b.reshape(b.size(0),-1).transpose(0,1))
                self.base5[('regular','trivial')]=b.transpose(0,1).reshape(shape)
            else:
                raise Exception(in_rep +' or '+ out + 'is not a right type')

        return self.base5[(in_rep,out_rep)]


    def base1x1(self, in_rep, out_rep):
        '''
            in: string indicate the representation type of input
            out: string indicate the representtation type of output
            ent the contant of scale transform. The group element will be of the form t^(-1), 1, t, t^2
        '''
        if(self.base1[(in_rep,out_rep)] is None):
            if(in_rep=='trivial' and out_rep=='regular'):
                base=[]
                s=torch.arange(self.n2).to(torch.float)
                for i in range(5, 6):
                    bas=torch.einsum('kijmn,t->ktijmn',self.rot_group.base(i, in_rep, out_rep), self.t**((i%5)*s))
                    base.append(bas.reshape(bas.size(0),bas.size(1)*bas.size(2), bas.size(3), bas.size(4), bas.size(5)))
                base=torch.cat(base, dim=0)
                shape=base.shape
                base,_=torch.qr(base.reshape(shape[0],-1).transpose(0,1))
                base=base.transpose(0,1).reshape(shape)
                self.base5[(in_rep,out_rep)]=base[::,::,::,2:3,2:3]
            elif(in_rep=='regular' and out_rep=='regular'):
                base=[]
                n=self.n2*2-1
                s=torch.zeros(self.n2, self.n2, n)
                for i in range(self.n2):
                    s[i,::,(self.n2-1-i):n-i]=torch.eye(self.n2)
                s1=torch.arange(self.n2).to(torch.float)
                for i in range(5,6):
                    bas=torch.torch.einsum('kijmn, t, tpq->qktipjmn',self.rot_group.base(i, in_rep, out_rep), self.t**((i%5)*s1), s)
                    a1,a2,a3,a4,a5,a6,a7,a8=bas.size()
                    base.append(bas.reshape(a1*a2, a3*a4, a5*a6, a7, a8))
                base=torch.cat(base, dim=0)
                shape=base.shape
                base,_=torch.qr(base.reshape(shape[0],-1).transpose(0,1))
                base=base.transpose(0,1).reshape(shape)
                self.base5[(in_rep,out_rep)]=base[::,::,::,2:3,2:3]
            elif(in_rep=='regular' and out_rep=='trivial'):
                b=torch.einsum('pm,qn,ij->pqijmn', torch.eye(1), torch.eye(1), torch.ones(1, self.dim))
                a1,a2,a3,a4,a5,a6=b.shape
                b=b.reshape(a1*a2,a3,a4,a5,a6)
                shape=b.shape
                b,_=torch.qr(b.reshape(b.size(0),-1).transpose(0,1))
                self.base5[('regular','trivial')]=b.transpose(0,1).reshape(shape)

        return self.base5[(in_rep,out_rep)]

    def conv(self, in_type, out_type, kernel_size=5, stride=1, bias=False):
        in_rep, dim_in=in_type
        out_rep, dim_out=out_type
        if(kernel_size==5):
            base=self.base5x5(in_rep, out_rep)
        elif(kernel_size==1):
            base=self.base1x1(in_rep, out_rep)

        return conv_base(base, dim_in, dim_out, stride, bias)

    def norm(self, in_type, affine=True, momentum=0.1, track_running_stats=True):
        '''
            num_rep: number of representation in the input
            rep: a string indicate the type of representation
            other argument is the same with the standard BatchNorm
        '''
        rep, num_rep=in_type
        if(rep=='regular'):
            dim_rep=self.dim
        elif(rep=='trivial'):
            dim_rep=1
        return GroupBatchNorm(num_rep, dim_rep, affine, momentum, track_running_stats)

    def GroupPool(self, in_type):
        rep, num_rep=in_type
        return GroupPooling(self.rep[rep], num_rep)

    def MaxPool(self, in_type, kernel_size=5):
        rep, num_rep=in_type
        C=self.rep[rep]*num_rep
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), Downsample(channels=C, filt_size=kernel_size, stride=2))

    def AvgPool(self, in_type, kernel_size=5 ):
        rep, num_rep=in_type
        C=self.rep[rep]*num_rep
        return nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1), Downsample(channels=C, filt_size=kernel_size, stride=2))




    
g=SR_group(4, False, 4, 0.9)
# print(g.base5x5('regular', 'regular').shape)
a=torch.randn(1,16,32,32)
net=g.conv(('regular',1), ('regular',2),5)
net(a)
# b=g.rot_group.GroupRotate(net(a))
# print(torch.sum(b**2)/(b.size(1)*b.size(2)*b.size(0)*b.size(3)))
# a1=g.rot_group.GroupRotate(a)
# b1=net(a1)
# print(torch.sum((b1-b)**2))



