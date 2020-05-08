import math
import torch
from utils import *
# from e2cnn import gspaces
# from e2cnn import nn
# N=4
# r2_act = gspaces.Rot2dOnR2(N=N)
# n=100
# feat_type_in = nn.FieldType(r2_act, [r2_act.regular_repr]*n)

# feat_type_out = nn.FieldType(r2_act, [r2_act.regular_repr])

# feat_type_out = nn.FieldType(r2_act, [r2_act.regular_repr])
# conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=1)

# x = torch.randn(1, N*n, 32, 32)
# x = nn.GeometricTensor(x, feat_type_in)
# y=conv(x).tensor
# print(torch.sum(y**2)/(y.size(0)*y.size(1)*y.size(2)*y.size(3)))
# print(torch.sum(conv.weights**2))
# # print(conv.basis_filter)



g=Group(4,True)
# b=g.base(4,'regular','regular')

# base=b.size(0)
num_in=10000
num_out=1
in_type=['regular',num_in]
out_type=['regular',num_out]
# num_out=1
# x=torch.randn(1,num_in*g.rep['regular'].dim,30,30)
# # w=torch.randn(num_in,num_out,base)
# w=kaiming_init(b,num_in,num_out)
# kernel=torch.einsum('ijk,kmnpq->jminpq',w, b)\
#             .reshape(num_out*g.rep['regular'].dim,num_in*g.rep['regular'].dim,5,5)
# kernel1=math.sqrt(2/(num_in*g.rep['regular'].dim*25))*torch.randn(num_out*g.rep['regular'].dim,num_in*g.rep['regular'].dim,5,5)
# print('ok',(torch.sum(x[0,::,0:5,0:5]*kernel1))**2)
# net1=torch.nn.Conv2d(num_in*g.rep['regular'].dim,num_out*g.rep['regular'].dim,5,padding=2)
# # torch.nn.init.kaiming_normal(net1.weight)
# net.weight=torch.nn.Parameter(kernel1)
# y1=net1(x)
# y=torch.nn.functional.conv2d(x,kernel,bias=None,stride=1,padding=2)
# print(torch.sum((y)**2)/(y.size(0)*y.size(1)*y.size(2)*y.size(3)))
# print(torch.sum((y1)**2)/(y1.size(0)*y1.size(1)*y1.size(2)*y1.size(3)))


conv=g.conv_fast(in_type,out_type)
x=torch.randn(1, g.rep['regular'].dim*num_in,30,30)
y=conv(x)
print(torch.sum((y)**2)/(y.size(0)*y.size(1)*y.size(2)*y.size(3)))


# a=torch.zeros(3,3)
# a[1,1]=1
# print(a)
# x=torch.ones(5,5).reshape(1,1,5,5)
# print(x)
# a=a.reshape(1,1,3,3)
# y=torch.nn.functional.conv2d(x,a,padding=1)
# print(y)