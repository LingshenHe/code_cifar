import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as la
import time
import math
from .utils import *
import torch.nn.functional as F
__all__=['SR_wrn28_10_d8d4d4', 'SR_wrn28_10_d8d8d8','SR_wrn28_10_d8d4d1', 'SR_wrn28_10_d8d8d4']



def regular_feature_type(group, planes: int, fixparams: bool = False):
    """ build a regular feature map with the specified number of channels"""
    

    N = group.dim
    
    if fixparams:
        planes *= math.sqrt(N)
    
    planes = 2*planes / N
    planes = int(planes)
    
    return ['regular', planes]


def trivial_feature_type(group, planes: int, fixparams: bool = False):
    """ build a trivial feature map with the specified number of channels"""
    
    N=group.dim
    if fixparams:
        planes *= math.sqrt(N)
        
    planes = int(planes)
    return ['trivial', planes]


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}



class WideBasic(nn.Module):
    
    def __init__(self,
                 in_type,
                 inner_type,
                 group,
                 dropout_rate: float,
                 stride: int = 1,
                 out_type = None,
                 ):
        super(WideBasic, self).__init__()
        
        if out_type is None:
            out_type = in_type
        
        self.in_type = in_type
        inner_type = inner_type
        self.out_type = out_type
        
        
        self.bn1 = group.norm(self.in_type)
        self.relu1 = nn.ReLU( inplace=True)
        if(group.n<=4):
            self.conv1 = group.conv3x3(self.in_type, inner_type)
        else:
            self.conv1 = group.conv5x5(self.in_type, inner_type)
        
        self.bn2 = group.norm(inner_type)
        self.relu2 = nn.ReLU( inplace=True)
        self.drop=dropout_rate
        if(group.n<=4):
            self.conv2 = group.conv3x3( inner_type, self.out_type, stride=stride)
        else:
            self.conv2= group.conv5x5(inner_type, self.out_type, stride=stride)
        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            # self.shortcut = group.conv([0], self.in_type, self.out_type, stride=stride)
            self.shortcut = group.conv1x1(self.in_type, self.out_type, stride=stride)
    
    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x

        return out
    
    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class SR_Wide_ResNet(torch.nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=10,
                 N: int = 8,
                 r: int = 1,
                 f: bool = True,
                 scale: bool = False,
                 deltaorth: bool = False,
                 fixparams: bool = False,
                 initial_stride: int = 1,
                 ):
        r"""
        
        Build and equivariant Wide ResNet.
        
        The parameter ``N`` controls rotation equivariance and the parameter ``f`` reflection equivariance.
        
        More precisely, ``N`` is the number of discrete rotations the model is initially equivariant to.
        ``N = 1`` means the model is only reflection equivariant from the beginning.
        
        ``f`` is a boolean flag specifying whether the model should be reflection equivariant or not.
        If it is ``False``, the model is not reflection equivariant.
        
        ``r`` is the restriction level:
        
        - ``0``: no restriction. The model is equivariant to ``N`` rotations from the input to the output
        
        - ``1``: restriction before the last block. The model is equivariant to ``N`` rotations before the last block
               (i.e. in the first 2 blocks). Then it is restricted to ``N/2`` rotations until the output.
        
        - ``2``: restriction after the first block. The model is equivariant to ``N`` rotations in the first block.
               Then it is restricted to ``N/2`` rotations until the output (i.e. in the last 3 blocks).
        
        - ``3``: restriction after the first and the second block. The model is equivariant to ``N`` rotations in the first
               block. It is restricted to ``N/2`` rotations before the second block and to ``1`` rotations before the last
               block.
        
        NOTICE: if restriction to ``N/2`` is performed, ``N`` needs to be even!
        
        """
        super(SR_Wide_ResNet, self).__init__()
        
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        
        print(f'| Wide-Resnet {depth}x{k}')
        
        nStages = [16, 16 * k, 32 * k, 64 * k]
        

        self._fixparams = fixparams
        
        self._layer = 0
        
        # number of discrete rotations to be equivariant to
        self._N = N
        
        # if the model is [F]lip equivariant
        self._f = f
        
        self.grouplist=[]
        
        self.grouplist.append(Group(N, f))


        # level of [R]estriction:
        #   r = 0: never do restriction, i.e. initial group (either DN or CN) preserved for the whole network
        #   r = 1: restrict before the last block, i.e. initial group (either DN or CN) preserved for the first
        #          2 blocks, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the last block
        #   r = 2: restrict after the first block, i.e. initial group (either DN or CN) preserved for the first
        #          block, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the last 2 blocks
        #   r = 3: restrict after each block. Initial group (either DN or CN) preserved for the first
        #          block, then restrict to N/2 rotations (either D{N/2} or C{N/2}) in the second block and to 1 rotation
        #          in the last one (D1 or C1)
        assert r in [0, 1, 2, 3]
        self._r = r
        
        # the input has 3 color channels (RGB).
        # Color channels are trivial fields and don't transform when the input is rotated or flipped
        r1 = ['trivial',3]
        
        # input field type of the model
        self.in_type = r1
        
        # in the first layer we always scale up the output channels to allow for enough independent filters
        r2 = FIELD_TYPE["regular"](self.grouplist[0], nStages[0], fixparams=True)
        
        # dummy attribute keeping track of the output field type of the last submodule built, i.e. the input field type of
        # the next submodule to build
        self._in_type = r2
        
        self.conv1 = self.grouplist[0].conv5x5( r1, r2)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=initial_stride)
        if self._r >= 2:
            
            self.restrict1 = self.grouplist[0].Restrictlayer(self._in_type)
            self._in_type=[self._in_type[0], self._in_type[1]*2]
            self.grouplist.append(Group(self._N//2, self._f))
            self._N=self._N//2
        else:
            self.restrict1 = lambda x: x
        
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        if self._r == 3:
        
            self.restrict2 = nn.Sequential(self.grouplist[-1].Restrictlayer(self._in_type), GroupRestrict(2,self._in_type[1]*2,True))
            self._in_type=['regular', 4*self._in_type[1]]
            self.grouplist.append(Group(self._N//4, self._f))
            self._N=self._N//4
        elif self._r == 1:
            self.restrict2 = self.grouplist[0].Restrictlayer(self._in_type)
            self._in_type=[self._in_type[0], self._in_type[1]*2]
            self.grouplist.append(Group(self._N//2, self._f))
            self._N=self._N//2
        else:
            self.restrict2 = lambda x: x
        
        # last layer maps to a trivial (invariant) feature map
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2, totrivial=True)
        # self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.pool= self.grouplist[-1].GroupPool(self._in_type)
        self._in_type=['trivial', self._in_type[1]]
        self.bn = self.grouplist[-1].norm(self._in_type, momentum=0.1)
        self.relu = nn.ReLU( inplace=True)
        
        self.linear = torch.nn.Linear(self._in_type[1]*self.grouplist[-1].rep[self._in_type[0]].dim, num_classes)
        # for name, module in self.named_modules():
        #     if isinstance(module, enn.R2Conv):
        #         if deltaorth:
        #             init.deltaorthonormal_init(module.weights, module.basisexpansion)
        #     elif isinstance(module, torch.nn.BatchNorm2d):
        #         module.weight.data.fill_(1)
        #         module.bias.data.zero_()
        #     elif isinstance(module, torch.nn.Linear):
        #         module.bias.data.zero_()
        
        print("MODEL TOPOLOGY:")
        for i, (name, mod) in enumerate(self.named_modules()):
            print(f"\t{i} - {name}")
    
   
        
    
    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    totrivial: bool = False
                    ):
    
        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        main_type = FIELD_TYPE["regular"](self.grouplist[-1], planes, fixparams=self._fixparams)
        inner_type = FIELD_TYPE["regular"](self.grouplist[-1], planes, fixparams=self._fixparams)
        
        if totrivial:
            out_type = FIELD_TYPE["regular"](self.grouplist[-1], planes*2, fixparams=self._fixparams)
        else:
            out_type = FIELD_TYPE["regular"](self.grouplist[-1], planes, fixparams=self._fixparams)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(block(self._in_type, inner_type, self.grouplist[-1], dropout_rate, stride, out_type=out_f))
            self._in_type = out_f
            
        print("layer", self._layer, "built")
        return nn.Sequential(*layers)
    
    def features(self, x):
        
        
        out = self.conv1(x)
        
        x1 = self.layer1(out)
        
        x2 = self.layer2(self.restrict1(x1))
        
        x3 = self.layer3(self.restrict2(x2))
        
        return x1, x2, x3
    
    def forward(self, x):

        # wrap the input tensor in a GeometricTensor
        
        out = self.conv1(x)
        out = self.layer1(out)
        
        out = self.layer2(self.restrict1(out))
        
        out = self.layer3(self.restrict2(out))
        out = self.pool(out)
        out = self.bn(out)
        out = self.relu(out)
        
        
        # extract the tensor from the GeometricTensor to use the common Pytorch operations
        # print(self._in_type[1]*self.grouplist[-1].dim)

        
        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

    # def cuda0(self):
    #     for i in self.grouplist:
    #         i.cuda()
    #     self.cuda()
    
def SR_wrn28_10_d8d4d4(**kwargs):
    """Constructs a Wide ResNet 28-10 model

    The model's block are respectively D8, D4 and D4 equivariant.

    """
    return SR_Wide_ResNet(28, 10, initial_stride=1, N=8, f=True, r=2, **kwargs)

def SR_wrn28_10_d8d8d8(**kwargs):
    """Constructs a Wide ResNet 28-10 model

    The model's block are respectively D8, D8 and D8 equivariant.

    """
    return SR_Wide_ResNet(28, 10, initial_stride=1, N=8, f=True, r=0, **kwargs)


def SR_wrn28_10_d8d4d1(**kwargs):
    """Constructs a Wide ResNet 28-10 model

    The model's block are respectively D8, D4 and D1 equivariant.

    """
    return SR_Wide_ResNet(28, 10, initial_stride=1, N=8, f=True, r=3, **kwargs)


def SR_wrn28_10_d8d8d4(**kwargs):
    """Constructs a Wide ResNet 28-10 model

    The model's block are respectively D8, D4 and D1 equivariant.

    """
    return SR_Wide_ResNet(28, 10, initial_stride=1, N=8, f=True, r=1, **kwargs)


def test():
    net=SR_wrn28_10_d8d8d4(dropout_rate=0.)
    # model_parameters = filter(lambda p: p.requires_grad, net.conv1.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    # model_parameters = filter(lambda p: p.requires_grad, net.layer1.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    # model_parameters = filter(lambda p: p.requires_grad, net.layer2.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    print(len(net.grouplist))
    
#     # class MNISTMODEL()

# net=SR_wrn28_10_d8d4d1(dropout_rate=0.)
# test()
# x=torch.randn(2,3,28,28)
# net=SR_wrn28_10_d8d4d4(dropout_rate=0.)
# y=net(x)
test()


