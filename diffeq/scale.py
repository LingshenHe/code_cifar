import torch
import numpy
import torch.nn as nn
import math
from anti import Downsample


class conv1(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size=3, stride=1, bias=False):
        super(conv1, self).__init__()
        ker=[]
        ker.append(torch.tensor([[-0.5,0.,0.5]]))
        ker.append(torch.tensor([[1.,-2.,1.]]))
        ker.append(torch.tensor([[0.5],[0.],[-0.5]]))
        ker.append(torch.tensor([[1.],[-2.],[1.]]))
        a=torch.ones(input_channel, 1).to(torch.float)
        self.stride=stride
        self.padding=padding=int(kernel_size/2)
        self.input_channel=input_channel
        self.out_channel=out_channel
        ker_=[None]*4
        self.kernel_size=kernel_size
        ker_[0]=torch.einsum('ij,mn->ijmn', a, ker[0])#ux
        ker_[1]=torch.einsum('ij,mn->ijmn', a, ker[1])#uxx
        ker_[2]=torch.einsum('ij,mn->ijmn', a, ker[2])#uy
        ker_[3]=torch.einsum('ij,mn->ijmn', a, ker[3])#uyy
        self.kernel_1=nn.Parameter(torch.cat(ker_[0:2], dim=0),requires_grad=False)
        self.kernel_2=nn.Parameter(torch.cat(ker_[2:4], dim=0),requires_grad=False)
        self.kernel_3=nn.Parameter(torch.cat([ker_[1], ker_[0]],dim=0),requires_grad=False)
        self.conv=nn.Conv2d(self.input_channel*9, self.out_channel, 1, stride=stride, bias=bias)
        self.norm=nn.GroupNorm(1,self.out_channel, affine=False)
        nn.init.kaiming_normal(self.conv.weight)
    
    
    def forward(self, x):
        x0=nn.functional.conv2d(x, self.kernel_1, stride=1, padding=(0,1),groups=self.input_channel)#[ux,uxx]
        x1=nn.functional.conv2d(x, self.kernel_2, stride=1, padding=(1,0),groups=self.input_channel)#[uy,uyy]
        x2=nn.functional.conv2d(x0, self.kernel_2, stride=1, padding=(1,0), groups=2*self.input_channel)#[uxy,uxxyy]
        x3=nn.functional.conv2d(x1, self.kernel_3, stride=1, padding=(0,1), groups=2*self.input_channel)#[uxxy,uxyy]
        x=torch.cat([x,x0[::,0:self.input_channel],x1[::,0:self.input_channel]\
        ,x0[::,self.input_channel:],x1[::,self.input_channel:],x2[::,0:self.input_channel],x3,x2[::,self.input_channel:]], dim=1)
        x=self.conv(x)
        return x

class conv2(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size=3, stride=1, bias=False):
        super(conv2, self).__init__()
        ker=torch.tensor([[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],[[0.,0.,0.],[-0.5,0.,0.5],[0.,0.,0.]],\
            [[0.,0.5,0.],[0.,0.,0.],[0.,-0.5,0.]],\
            [[0.,0.,0.],[1.,-2.,1.],[0.,0.,0.]],[[0.,1.,0.],[0.,-2.,0.],[0.,1.,0.]],\
                [[-0.25,0.,0.25],[0.,0.,0.],[0.25,0.,-0.25]],[[0.5,-1.,0.5],[0.,0.,0.],[-0.5,1.,-0.5]],\
                    [[-0.5,0.,0.5],[1.,0.,-1.],[-0.5,0.,0.5]],[[1.,-2.,1.],[-2.,4.,-2.],[1.,-2.,1.]]
            ])
        for i in range(ker.size(0)):
            ker[i]=ker[i]/torch.sum(ker[i]**2).sqrt()
        self.ker=nn.Parameter(ker,requires_grad=False)
        self.weight=nn.Parameter((1/(3*math.sqrt(input_channel)))*torch.randn(out_channel, input_channel, 9))
        self.bias=None
        if(bias==True):
            self.bias=nn.Parameter(torch.zeros(out_channel))
        self.stride=stride
    def forward(self, x):
        kernel=torch.einsum('ijk,kmn->ijmn', self.weight, self.ker)
        if(self.bias is not None):
            return nn.functional.conv2d(x, kernel, bias=self.bias, stride=self.stride, padding=1)
        else:
            return nn.functional.conv2d(x, kernel, stride=self.stride, padding=1)

class conv3(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size=3, stride=1, bias=True):
        super(conv3, self).__init__()
        ker=torch.tensor([[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],[[0.,0.,0.],[-0.5,0.,0.5],[0.,0.,0.]],\
            [[0.,0.5,0.],[0.,0.,0.],[0.,-0.5,0.]],\
            [[0.,0.,0.],[1.,-2.,1.],[0.,0.,0.]],[[0.,1.,0.],[0.,-2.,0.],[0.,1.,0.]],\
                [[-0.25,0.,0.25],[0.,0.,0.],[0.25,0.,-0.25]],[[0.5,-1.,0.5],[0.,0.,0.],[-0.5,1.,-0.5]],\
                    [[-0.5,0.,0.5],[1.,0.,-1.],[-0.5,0.,0.5]],[[1.,-2.,1.],[-2.,4.,-2.],[1.,-2.,1.]]
            ])
        self.input_channel=input_channel
        self.out_channel=out_channel
        for i in range(ker.size(0)):
            ker[i]=ker[i]/torch.sum(ker[i]**2).sqrt()
        a=torch.ones(input_channel).to(torch.float)
        self.ker=nn.Parameter(torch.einsum('k,jmn->kjmn',a, ker).reshape(9*input_channel,1,3,3),requires_grad=False)
        self.weight=nn.Parameter((1/(3*math.sqrt(input_channel)))*torch.randn(out_channel, 9*input_channel, 1, 1))
        self.bias=None
        if(bias==True):
            self.bias=nn.Parameter(torch.zeros(out_channel))
        self.stride=stride
        self.norm=nn.GroupNorm(9,9*input_channel,affine=False)
    def forward(self, x):
        x1=nn.functional.conv2d(x, self.ker, stride=1, padding=1, groups=self.input_channel)
        if(self.bias is not None):
            return nn.functional.conv2d(x1, self.weight, bias=self.bias, stride=self.stride, padding=0)
        else:
            return nn.functional.conv2d(x1, self.weight, stride=self.stride, padding=0)

class conv4(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size=3, stride=1, bias=True):
        super(conv4, self).__init__()
        ker=torch.tensor([[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],[[0.,0.,0.],[-0.5,0.,0.5],[0.,0.,0.]],\
            [[0.,0.5,0.],[0.,0.,0.],[0.,-0.5,0.]],\
            [[0.,0.,0.],[1.,-2.,1.],[0.,0.,0.]],[[0.,1.,0.],[0.,-2.,0.],[0.,1.,0.]],\
                [[-0.25,0.,0.25],[0.,0.,0.],[0.25,0.,-0.25]],[[0.5,-1.,0.5],[0.,0.,0.],[-0.5,1.,-0.5]],\
                    [[-0.5,0.,0.5],[1.,0.,-1.],[-0.5,0.,0.5]],[[1.,-2.,1.],[-2.,4.,-2.],[1.,-2.,1.]]
            ])
        self.input_channel=input_channel
        self.out_channel=out_channel
        for i in range(ker.size(0)):
            ker[i]=ker[i]/torch.sum(ker[i]**2).sqrt()
        a=torch.ones(input_channel).to(torch.float)
        self.ker=nn.Parameter(ker.reshape(9,1,1,3,3),requires_grad=False)
        self.weight=nn.Parameter((1/(3*math.sqrt(input_channel)))*torch.randn(out_channel, 9*input_channel, 1, 1))
        self.bias=None
        if(bias==True):
            self.bias=nn.Parameter(torch.zeros(out_channel))
        self.stride=stride
        self.norm=nn.GroupNorm(1,9*input_channel,affine=False)
    def forward(self, x):
        b,c,w,h=x.size()
        x1=nn.functional.conv3d(x.reshape(b,1,c,w,h), self.ker, stride=1, padding=(0,1,1)).reshape(b,9*c,w,h)
        x1=self.norm(x1)
        if(self.bias is not None):
            return nn.functional.conv2d(x1, self.weight, bias=self.bias, stride=self.stride, padding=0)
        else:
            return nn.functional.conv2d(x1, self.weight, stride=self.stride, padding=0)


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

class conv1x1(nn.Module):
    def __init__(self, n, num_in, num_out, stride, bias):
        '''
        n: number of group element in the scale group
        
        '''
        super(conv1x1, self).__init__()
        self.n=n
        n=2*n-1
        self.num_in=num_in
        self.num_out=num_out
        self.stride=stride
        s=torch.zeros(self.n, self.n, n)
        for i in range(self.n):
            s[i,::,(self.n-1-i):n-i]=torch.eye(self.n)
        self.s=nn.Parameter(s, requires_grad=False)
        #kaiming initialization
        self.weight=nn.Parameter((1/math.sqrt(self.n*num_in))*torch.randn(num_out, num_in, n))
        self.bias=None
        if(bias==True):
            self.bias=nn.Parameter(torch.zeros(self.num_out), requires_grad=True)
            self.a=nn.Parameter(torch.ones(self.dim_rep_in), requires_grad=False)
        
    def forward(self, x):
        bias=None
        kernel=torch.einsum('ijn,pqn->piqj', self.s, self.weight).reshape(self.num_out*self.n, self.num_in*self.n,1,1)
        if self.bias is not None:
            bias=torch.einsum('i,j->ij', self.bias, self.a).reshape(-1)
        return nn.functional.conv2d(x,kernel,bias=bias, stride=self.stride, padding=0)




class scale_group:
    def __init__(self, n, t):
        self.n=n
        self.t=t
        self.bases={('trivial','regular'):None, ('regular','regular'):None, ('regular','trivial'):None}
        self.a=torch.ones(n,9)
        self.coef=torch.tensor([0.,1.,1.,2.,2.,2.,3.,3.,4.])
        for i in range(n):
            for j in range(9):
                self.a[i,j]=t**(i*self.coef[j])
        self.ker=torch.tensor([[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],[[0.,0.,0.],[-0.5,0.,0.5],[0.,0.,0.]],\
            [[0.,0.5,0.],[0.,0.,0.],[0.,-0.5,0.]],\
            [[0.,0.,0.],[1.,-2.,1.],[0.,0.,0.]],[[0.,1.,0.],[0.,-2.,0.],[0.,1.,0.]],\
                [[-0.25,0.,0.25],[0.,0.,0.],[0.25,0.,-0.25]],[[0.5,-1.,0.5],[0.,0.,0.],[-0.5,1.,-0.5]],\
                    [[-0.5,0.,0.5],[1.,0.,-1.],[-0.5,0.,0.5]],[[1.,-2.,1.],[-2.,4.,-2.],[1.,-2.,1.]]
            ])
        self.rep={'trivial':1, 'regular':n}


    def base(self, in_rep, out,):
        '''
            in: string indicate the representation type of input
            out: string indicate the representtation type of output
            n: represent the number of group elements
            t: represent the contant of scale transform. The group element will be of the form t^(-1), 1, t, t^2
        '''
        if(in_rep=='trivial' and out=='regular'):
            if(self.bases[('trivial','regular')] is not None):
                return self.bases[('trivial','regular')]
            b=torch.einsum('ikq,kmn->kiqmn', self.a.reshape(self.n, 9, 1), self.ker)
            shape=b.shape
            b,_=torch.qr(b.reshape(b.size(0),-1).transpose(0,1))
            self.bases[('trivial','regular')]=b.transpose(0,1).reshape(shape)
            return self.bases[('trivial','regular')]
        elif (in_rep=='regular' and out=='regular'):
            n=self.n*2-1
            if(self.bases[('regular','regular')] is not None):
                return self.bases[('regular','regular')]
            s=torch.zeros(self.n, self.n, n)
            for i in range(self.n):
                s[i,::,(self.n-1-i):n-i]=torch.eye(self.n)
            b=torch.einsum('ijp,ik,kmn->pkijmn', s, self.a, self.ker)
            # print(b[0,1])
            a1,a2,a3,a4,a5,a6=b.shape
            b=b.reshape(a1*a2,a3,a4,a5,a6)
            shape=b.shape
            b=b.reshape(b.size(0),-1).transpose(0,1)
            b,_=torch.qr(b)
            self.bases[('regular','regular')]=b.transpose(0,1).reshape(shape)
            # print(self.bases[('regular','regular')][15])
            return self.bases[('regular','regular')]
        
        elif (in_rep=='regular' and out=='trivial'):
            if(self.bases[('regular','trivial')] is not None):
                return self.bases[('regular','trivial')]
            b=torch.einsum('pm,qn,ij->pqijmn', torch.eye(3), torch.eye(3), torch.ones(1, self.n))
            a1,a2,a3,a4,a5,a6=b.shape
            b=b.reshape(a1*a2,a3,a4,a5,a6)
            shape=b.shape
            b,_=torch.qr(b.reshape(b.size(0),-1).transpose(0,1))
            self.bases[('regular','trivial')]=b.transpose(0,1).reshape(shape)
            return self.bases[('regular','trivial')]
        else:
            raise Exception(in_rep +' or '+ out + 'is not a right type')
        
    def conv5x5(self, in_type, out_type, stride=1, bias=False):
        in_rep, dim_in=in_type
        out_rep, dim_out=out_type
        base=self.base(in_rep, out_rep)

        return conv_base(base, dim_in, dim_out, stride, bias)

    def conv1x1(self, in_type, out_type, stride=1, bias=False):
        in_rep, dim_in=in_type
        out_rep, dim_out=out_type
        return conv1x1(self.n, dim_in, dim_out, stride, bias)

    def norm(self, in_type, affine=True, momentum=0.1, track_running_stats=True):
        '''
            num_rep: number of representation in the input
            rep: a string indicate the type of representation
            other argument is the same with the standard BatchNorm
        '''
        rep, num_rep=in_type
        dim_rep=self.rep[rep]
        return GroupBatchNorm(num_rep, dim_rep, affine, momentum, track_running_stats)

    def GroupPool(self, in_type):
        rep, num_rep=in_type
        return GroupPooling(self.rep[rep], num_rep)

    def MaxPool(self, in_type, kernel_size=5):
        print(in_type)
        rep, num_rep=in_type
        C=self.rep[rep]*num_rep
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), Downsample(channels=C, filt_size=kernel_size, stride=2))

    def AvgPool(self, in_type, kernel_size=5 ):
        rep, num_rep=in_type
        C=self.rep[rep]*num_rep
        return nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1), Downsample(channels=C, filt_size=kernel_size, stride=2))











def conv(input_channel, out_channel, kernel_size, stride, bias):
    return conv4(input_channel, out_channel, kernel_size, stride, bias)
# a=torch.randn(1,100,32,32)
# net=conv4(100,5,stride=2)
# # net.eval()
# net1=conv2(100,5,stride=2)
# net1.weight=nn.Parameter(net.weight.reshape(5,9,100).transpose(1,2))
# # print(net1.weight[0,::,0])
# # print(net.weight[0,0:net.input_channel])
# b=net(a)
# b1=net1(a)
# # print(net.kernel_1)
# print(torch.sum(b**2)/(5*16*16))
# # print(b.shape)
# print(torch.sum((b-b1)**2))


# a=2*torch.randn(10,10,10,10)
# net=nn.GroupNorm(10,1,affine=False)
# y=net(a)
# print(torch.sum(y**2)/(10**4))


# x=torch.randn(1,8,32,32)
# w1=torch.randn(4,4,3,3)
# w2=torch.randn(4,4,3,3)
# w3=torch.cat([w1,w2],dim=0)
# y1=nn.functional.conv2d(x,w3,padding=1,groups=2)
# y2=nn.functional.conv2d(x[::,0:4],w1,padding=1)
# y3=nn.functional.conv2d(x[::,4:8],w2,padding=1)
# y4=torch.cat([y2,y3],dim=1)
# print(y4.shape)
# print(torch.sum(y4-y1))

# a=torch.arange(0,90,1)
# x=a.reshape(1,10,9)
# y=a.reshape(1,90,1,1)
# print(y.reshape(1,10,9)l;，，，。，；l)


# g=scale_group(4, 0.9)
# a=torch.randn(1,25,40,40)
# net=g.conv5x5(('trivial',25), ('regular',5))
# b=net(a)
# print(torch.sum(b**2)/(b.size(1)*b.size(2)*b.size(0)*b.size(3)))