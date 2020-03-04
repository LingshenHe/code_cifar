import torch
import e2cnn
import torch.nn as nn
import numpy as np
from numpy import linalg as la
import time
import math



def kronecker(x,y):
    return torch.einsum('ij,mn->imjn',[x,y]).reshape(x.size(0)*y.size(0),x.size(1)*y.size(1))



def interpolate(x,y):
    x1=math.floor(x)
    y1=math.floor(y)
    return [x1,x1+1,y1,y1+1]



def numpy_rotate(x,a):
    ### x is the input image of shape: bxcxhxw
    ### a is the rotational matrix
    height,width=x.shape[-2:]
    # print(height,width)
    cen=np.array([(height-1)/2.,(width-1)/2.])
    y=np.zeros_like(x)
    
    for i in range(height):
        for j in range(width):
            f=0
            t=np.array([i,j])
            c=np.matmul(la.inv(a),t-cen)+cen
            # print('c',c)
            z=interpolate(c[0],c[1])
            if(z[0]>=0 and z[0]<height and z[2]>=0 and z[2]<width):
                f=f+x[::,::,z[0],z[2]]*(z[1]-c[0])*(z[3]-c[1])
            if(z[0]>=0 and z[0]<height and z[3]>=0 and z[3]<width):
                f=f+x[::,::,z[0],z[3]]*(z[1]-c[0])*(c[1]-z[2])   
            if(z[1]>=0 and z[1]<height and z[2]>=0 and z[2]<width):
                f=f+x[::,::,z[1],z[2]]*(c[0]-z[0])*(z[3]-c[1]) 
            if(z[1]>=0 and z[1]<height and z[3]>=0 and z[3]<width):
                f=f+x[::,::,z[1],z[2]]*(c[0]-z[0])*(c[1]-z[2]) 
            y[::,::,i,j]=f
    return y




def rotate(x,a):
    ### x is the input image of shape: bxcxhxw
    ### a is the rotational matrix

    a=a.to(x.device)
    height,width=x.shape[-2:]
    cen=torch.tensor([(height-1)/2.,(width-1)/2.]).to(x.device)
    y=torch.zeros_like(x)
    
    for i in range(height):
        for j in range(width):
            f=torch.tensor([0.],device=x.device)
            t=torch.tensor([i,j],device=x.device).to(torch.float32)
            c=torch.matmul(torch.inverse(a),t-cen)+cen
            z=interpolate(c[0],c[1])
            if(z[0]>=0 and z[0]<height and z[2]>=0 and z[2]<width):
                f=f+x[::,::,z[0],z[2]]*(z[1]-c[0])*(z[3]-c[1])
            if(z[0]>=0 and z[0]<height and z[3]>=0 and z[3]<width):
                f=f+x[::,::,z[0],z[3]]*(z[1]-c[0])*(c[1]-z[2])   
            if(z[1]>=0 and z[1]<height and z[2]>=0 and z[2]<width):
                f=f+x[::,::,z[1],z[2]]*(c[0]-z[0])*(z[3]-c[1]) 
            if(z[1]>=0 and z[1]<height and z[3]>=0 and z[3]<width):
                f=f+x[::,::,z[1],z[2]]*(c[0]-z[0])*(c[1]-z[2]) 
            y[::,::,i,j]=f
    return y



def rotating(t):
    '''
    t: the rotation angle
    return the rotation matrix
    '''
    t=np.pi*t/180.
    A=torch.zeros(2,2)
    A[0,0]=np.cos(t)
    A[0,1]=np.sin(t)
    A[1,0]=-np.sin(t)
    A[1,1]=np.cos(t)
    return A




# def scale():
#     #TODO



# x=np.arange(25)
# x=x.reshape(1,1,5,5)
# print(x)
# a=np.array([[-1.,0.],[0.,1.]])
# print(numpy_rotate(x,a))





class diff_rep:
    ### n represent the order of the rotation
    def __init__(self,n,flip=False):
      
        self.rep_e=[torch.tensor([[1.]])]
        self.rep_m=[torch.tensor([[1.]])]
        self.order=1
        self.n=n
        t=-2*np.pi/n
        self.flip=flip
        self.g_e=torch.tensor([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
        if(flip==True):
            self.g_m=torch.tensor([[1.,0.],[0.,-1.]])
            self.rep_m.append(self.g_m)
        self.rep_e.append(self.g_e)
    
    
     
    def next(self):
        if(self.order==4):
            print('Order is less than 5')
        a=self.rep_e[self.order]
        self.order=self.order+1
        b=torch.zeros(self.order+1,self.order+1)
        for i in range(self.order):
            for j in range(1,self.order):
                b[j,i]=self.g_e[0,0]*a[j,i]+self.g_e[1,0]*a[j-1,i]
            b[0,i]=self.g_e[0,0]*a[0,i]
            b[self.order,i]=self.g_e[1,0]*a[self.order-1,i]
        for j in range(1,self.order):
            b[j,self.order]=self.g_e[0,1]*a[j,i]+self.g_e[1,1]*a[j-1,i]
        b[0,self.order]=self.g_e[0,1]*a[0,i]
        b[self.order,self.order]=self.g_e[1,1]*a[self.order-1,i]
        self.rep_e.append(b)
        if(self.flip==True):
            a=self.rep_m[self.order-1]
            b=torch.zeros(self.order+1,self.order+1)
            for i in range(self.order):
                for j in range(1,self.order):
                    b[j,i]=self.g_m[0,0]*a[j,i]+self.g_m[1,0]*a[j-1,i]
                b[0,i]=self.g_m[0,0]*a[0,i]
                b[self.order,i]=self.g_m[1,0]*a[self.order-1,i]
            for j in range(1,self.order):
                b[j,self.order]=self.g_m[0,1]*a[j,i]+self.g_m[1,1]*a[j-1,i]
            b[0,self.order]=self.g_m[0,1]*a[0,i]
            b[self.order,self.order]=self.g_m[1,1]*a[self.order-1,i]
            self.rep_m.append(b)
    
    
    
    def __getitem__(self,i):
        if(i>4):
            print('Order is less than 5')
        while(i>self.order):
            self.next()
        if(self.flip==True):
            return self.rep_e[i],self.rep_m[i]
        else:
            return self.rep_e[i]


class c_regular:
    def __init__(self,n):
        self.type='cn'
        self.n=n
        self.rep_e=torch.zeros(n,n)
        self.rep_e[1:n,0:n-1]=torch.eye(n-1)
        self.rep_e[0,n-1]=1.
        self.dim=n
    


class d_regular:
    def __init__(self,n):
        self.n=n
        self.type='dn'
        self.rep_e=torch.zeros(2*n,2*n)
        self.rep_e[1:n,0:n-1]=torch.eye(n-1)
        self.rep_e[0,n-1]=1.
        self.rep_e[n:2*n-1,n+1:2*n]=torch.eye(n-1)
        self.rep_e[2*n-1,n]=1.
        self.rep_m=torch.zeros(2*n,2*n)
        self.rep_m[0:n,n:2*n]=torch.eye(n)
        self.rep_m[n:2*n,0:n]=torch.eye(n)
        self.dim=2*n



class trivial:
    def __init__(self):
        self.dim=1
        self.type='trivial'
        self.rep_e=torch.eye(1)
        self.rep_m=torch.eye(1)


def solve(A):
    n=torch.matrix_rank(A)
    u,s,v=torch.svd(A,some=True)
    return v[::,n:A.size(1)]



def solve_(A):
    G=la.pinv(a)
    w=np.eye(A.shape[1])-np.matmul(G,A)
    n=la.matrix_rank(w)
    u,s,v=la.svd(w)
    return u[::,0:n]




def compute_base(A):
    n=torch.matrix_rank(A)
    u,s,v=torch.svd(A,some=True)
    return v[::,0:n].transpose(0,1)




class iden(nn.Module):
    def __init__(self):
        super(iden,self).__init__()

    
    
    def forward(self,x):
        return x




# class conv_base(nn.Module):
#     def __init__(self, Group, orderlist, in_rep, out_rep, num_in, num_out, stride=1):
#         ''' 

#             Group: group containing the base
#             bases: number of basis kernel in a specified type
#             param: parameter tensor of shape dim_in x dim_out x bases
#             base: bases of a specified type of kernel, shape of  bases x dim_rep_out x dim_rep_in x kernel_size x kernel_size
#         '''
#         super(conv_base,self).__init__()
#         self.Group=Group
#         self.in_rep=in_rep
#         self.out_rep=out_rep
#         self.num_in=num_in
#         self.num_out=num_out
#         self.orderlist=orderlist
#         self.base=[]
#         for i in orderlist:
#             self.base.append(self.Group.base(i, in_rep, out_rep))
#         self.base=nn.Parameter(torch.cat(self.base),requires_grad=False)        
#         self.dim_rep_in=self.base.size(2)
#         self.dim_rep_out=self.base.size(1)
#         self.bases=self.base.size(0)
#         self.param=torch.nn.Parameter(torch.randn(num_in, num_out, self.bases))
#         self.stride=stride
#         self.size=self.base.size(4)
       
    
#     def forward(self,x):
#         #get the kernel of the conv from the base
#         kernel=torch.einsum('ijk,kmnpq->jminpq',self.param, self.base)\
#             .reshape(self.num_out*self.dim_rep_out,self.num_in*self.dim_rep_in,self.size,self.size)
#         return nn.functional.conv2d(x,kernel,bias=None, stride=self.stride, padding=math.floor(self.size/2))



class conv(nn.Module):
    def __init__(self, Group, orderlist, in_rep, out_rep, num_in, num_out, stride, scale=False, group=None):
        '''
            orderlist: a list consist of order of differential operator
            scale: the net is scale equivariant or not.
            group: is the arguments num_groups in the GroupNorm, it is dim_in as default.
            exist: if exist = True, then the 0 order differential operator is in the conv.
            the other argument is the same with conv_base.
        '''
        super(conv,self).__init__()
        self.num_in=num_in
        self.num_out=num_out
        self.orderlist=orderlist
        self.in_rep=in_rep
        self.out_rep=out_rep
        list=[]
        for i in orderlist:
            list.append(conv_base(Group.base(i, in_rep, out_rep), num_in, num_out,  stride))
      
        self.convlist=nn.ModuleList(list)
        if (group is None):
            group=num_out
        if scale==False:
            self.gn=iden()
        else:
            self.gn=torch.nn.GroupNorm(group,Group.rep[out_rep].dim*num_out)
    
    
    def forward(self,x):
        ### I think the program is slow compare to the e2cnn is due to the for loop here
        y=self.gn(self.convlist[0](x))
        for i in range(1,len(self.convlist)):
            y=y+self.gn(self.convlist[i](x))
        return y
    



class conv_fast(nn.Module):
    def __init__(self, base, num_in, num_out, stride):
        super(conv_fast, self).__init__()
        self.base=torch.nn.Parameter(base, requires_grad= False)
        self.bases=self.base.size(0)
        self.size=self.base.size(4)
        self.stride=stride
        self.param=torch.nn.Parameter(kaiming_init(base, num_in, num_out))
        self.dim_rep_in=self.base.size(2)
        self.dim_rep_out=self.base.size(1)
        self.num_in=num_in
        self.num_out=num_out



    def forward(self,x):
            #get the kernel of the conv from the base
            kernel=torch.einsum('ijk,kmnpq->jminpq',self.param, self.base)\
                .reshape(self.num_out*self.dim_rep_out,self.num_in*self.dim_rep_in,self.size,self.size)
            print(kernel)
            return nn.functional.conv2d(x,kernel,bias=None, stride=self.stride, padding=math.floor(self.size/2))










class conv_base(nn.Module):
    def __init__(self, base, num_in, num_out, stride=1):
        ''' 

            Group: group containing the base
            bases: number of basis kernel in a specified type
            param: parameter tensor of shape dim_in x dim_out x bases
            base: bases of a specified type of kernel, shape of  bases x dim_rep_out x dim_rep_in x kernel_size x kernel_size
        '''
        super(conv_base,self).__init__()
        self.base=torch.nn.Parameter(base,requires_grad=False)
        self.num_in=num_in
        self.num_out=num_out
        # self.base=torch.nn.Parameter(self.Group.base(order, in_rep, out_rep),requires_grad=False)
        self.dim_rep_in=self.base.size(2)
        self.dim_rep_out=self.base.size(1)
        self.bases=self.base.size(0)
        self.param=torch.nn.Parameter(kaiming_init(base, num_in, num_out))
        self.stride=stride
        self.size=self.base.size(4)
        self.kernel=torch.einsum('ijk,kmnpq->jminpq',self.param, self.base)\
            .reshape(self.num_out*self.dim_rep_out,self.num_in*self.dim_rep_in,self.size,self.size)
    
    def forward(self,x):
        #get the kernel of the conv from the base
        kernel=torch.einsum('ijk,kmnpq->jminpq',self.param, self.base)\
            .reshape(self.num_out*self.dim_rep_out,self.num_in*self.dim_rep_in,self.size,self.size)
        return nn.functional.conv2d(x,kernel,bias=None, stride=self.stride, padding=math.floor(self.size/2))






# class GroupBatchNorm(nn.Module):
#     def __init__(self, num_rep, dim_rep, affine=True, track_running_stats=True):
#         super(GroupBatchNorm,self).__init__()
#         self.gn=nn.BatchNorm2d(num_rep*dim_rep, affine=affine, track_running_stats=track_running_stats)
#         self.weight=torch.nn.Parameter(torch.ones(num_rep,1))
#         self.bias=torch.nn.Parameter(torch.zeros(num_rep,1))
#         self.num_rep=num_rep
#         self.dim_rep=dim_rep
    
#     def forward(self, x):
#         self.gn.weight=torch.nn.Parameter(torch.cat([self.weight]*self.dim_rep, dim=1).reshape(-1))
#         self.gn.bias=torch.nn.Parameter(torch.cat([self.bias]*self.dim_rep,dim=1).reshape(-1))
#         return self.gn(x)



# class GroupBatchNorm(nn.Module):
#     def __init__(self, num_rep, dim_rep, affine=False, momentum=0.1, track_running_stats=True):
#         super(GroupBatchNorm,self).__init__()
#         self.momentum=momentum
#         self.gn=nn.BatchNorm2d(num_rep*dim_rep, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
#         self.weight=torch.nn.Parameter(torch.ones(num_rep,1))
#         self.bias=torch.nn.Parameter(torch.zeros(num_rep,1))
#         self.num_rep=num_rep
#         self.dim_rep=dim_rep
    
#     def forward(self, x):
#         x=self.gn(x)
#         weight=torch.cat([self.weight]*self.dim_rep, dim=1).reshape(1,self.dim_rep*self.num_rep,1,1)
#         bias=torch.cat([self.bias]*self.dim_rep,dim=1).reshape(1,self.dim_rep*self.num_rep,1,1)
#         x=weight*x+bias
#         return x



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





class GroupRestrict(nn.Module):
    def __init__(self, n, num_in_rep, flip=False):
        '''
            n: number of rotations in the group
            num_in_rep: number of regular representation in the input
            flip: is the group DN
        '''
        super(GroupRestrict,self).__init__()
        self.n=n
        self.num_in_rep=num_in_rep
        if (flip==True):
            n=2*n
        tr=torch.zeros(n,n)
        for i in range(n):
            if(i%2==0):
                tr[int(i/2),i]=1
            else:
                tr[int(n/2+(i-1)/2),i]=1
        self.dim=n
        self.param=torch.nn.Parameter(tr, False)
    
    
    def forward(self, x):
        
        x=torch.einsum('ij,bkjmn->bkimn', self.param, x.reshape(x.size(0), self.num_in_rep, self.dim, x.size(2), x.size(3)))\
            .reshape(x.shape)
        return x



# class GroupPooling(nn.Module):
#     def __init__(self, type)


def kaiming_init(base, num_in, num_out, normal=True):
    '''
        base: the base of the conv_base or conv_fast
        num_in: number of representation in of the input in conv_base of conv_fast
        num_out: number of representation in of the output in conv_base of conv_fast
        normal: using the normal or constant initialization
    '''
    f=torch.sum(base*base)*num_in/(base.size(1))
    print(f)
    if(normal==True):
        weight=torch.sqrt(1/f)*torch.randn(num_in, num_out, base.size(0))
    else:
        weight=torch.sqrt(12/f)*(torch.rand(num_in, num_out, base.size(0))-0.5)
    return weight











class Group:
    def __init__(self, n, flip=False):
        '''
            n: n means the group has n pure rotational element.
            flip: wether the group include flip element
        '''
        self.dif_ker={\
    '0':torch.tensor([[[1.]]]),\
    '1':torch.tensor([[[0.,0.,0.],[-0.5,0.,0.5],[0.,0.,0.]],[[0.,0.5,0.],[0.,0.,0.],[0.,-0.5,0.]]]),\
    '2':torch.tensor([[[0.,0.,0.],[1.,-2.,1.],[0.,0.,0.]],[[-0.25,0,0.25],[0.,0.,0.],[0.25,0.,-0.25]],[[0.,1.,0.],[0.,-2.,0.],[0.,1.,0.]]]),\
    '3':torch.tensor([[[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[-0.5,1.,0.,-1.,0.5],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]],\
        [[0.,0.,0.,0.,0.],[0.,0.5,-1.,0.5,0.],[0.,0.,0.,0.,0.],[0.,-0.5,1.,-0.5,0.],[0.,0.,0.,0.,0.]],\
        [[0.,0.,0.,0.,0.],[0.,-0.5,0.,0.5,0.],[0.,1.,0.,-1.,0.],[0.,-0.5,0.,0.5,0.],[0.,0.,0.,0.,0.]],\
        [[0.,0.,0.5,0.,0.],[0.,0.,-1.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,-0.5,0.,0.]]]),\
    '4':torch.tensor([\
        [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[1.,-4.,6.,-4.,1.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]],\
            [[0.,0.,0.,0.,0.],[-0.25,0.5,0.,-0.5,0.25],[0.,0.,0.,0.,0.],[0.25,-0.5,0.,0.5,-0.25],[0.,0.,0.,0.,0.]],\
                [[0.,0.,0.,0.,0.],[0.,1.,-2.,1.,0.],[0.,-2.,4.,-2.,0.],[0.,1.,-2.,1.,0.],[0.,0.,0.,0.,0.]],\
                [[0.,-0.25,0.,0.25,0.],[0.,0.5,0.,-0.5,0.],[0.,0.,0.,0.,0.],[0.,-0.5,0.,0.5,0.],[0.,0.25,0.,-0.25,0.]],\
                    [[0.,0.,1.,0.,0.],[0.,0.,-4.,0.,0.],[0.,0.,6.,0.,0.],[0.,0.,-4.,0.,0.],[0.,0.,1.,0.,0.]]]),\
    '5':torch.tensor([\
        [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]]),\
    '6':torch.tensor([\
        [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,-0.5,0.,0.5,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]],\
        [[0.,0.,0.,0.,0.],[0.,0.,0.5,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,-0.5,0.,0.],[0.,0.,0.,0.,0.]]]),\
    '7':torch.tensor([\
        [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,1.,-2.,1.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]],\
            [[0.,0.,0.,0.,0.],[0.,-0.25,0,0.25,0.],[0.,0.,0.,0.,0.],[0.,0.25,0.,-0.25,0.],[0.,0.,0.,0.,0.]],\
                [[0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,-2.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.]]])}
        self.filter={'5':torch.tensor([[1.]]),\
            '6':torch.tensor([[1.,0.],[0.,1.]])}
        self.filter['7']=torch.eye(3).to(torch.float)
        self.filter['3']=torch.eye(4).to(torch.float)
        self.filter['3'][0,0]=0.
        self.filter['3'][3,3]=0.
        self.filter['4']=torch.zeros(5,5).to(torch.float)
        self.filter['4'][2,2]=1
        self.n=n
        self.flip=flip
        self.dim=n
        if (flip==True):
            self.dim=self.dim*2
        self.diff_rep=diff_rep(n, flip)
        self.rep={'regular':None, 'trivial':trivial()}
        if(flip==True):
            self.rep['regular']=d_regular(n)
        else:
            self.rep['regular']=c_regular(n)
        self.bases=[]
        for i in range(8):
            self.bases.append({('regular','regular'):None, ('regular','trivial'):None, ('trivial','trivial'):None,('trivial','regular'):None})
        self.fast_base={('regular','regular'):None, ('regular','trivial'):None, ('trivial','trivial'):None,('trivial','regular'):None}
        self.base_3x3={('regular','regular'):None, ('regular','trivial'):None, ('trivial','trivial'):None,('trivial','regular'):None}
    def coef(self, in_rep, out_rep, df):
        d=kronecker(in_rep.transpose(0,1),df)
        n1=d.size(0)
        n2=out_rep.size(0)
        return kronecker(out_rep,torch.eye(n1))-kronecker(torch.eye(n2),d)

    
    def base(self, order, in_rep, out_rep):
        '''
            order: an integer representing the order of differential operator.
            in_rep: a string indicate the representation type of input feature
            out_rep: a string indicate the representation type of output feature
            return the bases of the interwiners. of shape (bases x dim_out_rep x dim_in_rep x kernel_size x kernel_size)
        '''
        if(self.bases[order][(in_rep,out_rep)] is not None):
            return self.bases[order][(in_rep,out_rep)]
        in_rep_=self.rep[in_rep]
        out_rep_=self.rep[out_rep]
        if(self.flip==True):
            df1,df2=self.diff_rep[order%5]
            w1=self.coef(in_rep_.rep_e,out_rep_.rep_e,df1)
            w2=self.coef(in_rep_.rep_m,out_rep_.rep_m,df2)
            w=torch.cat((w1,w2))
        else:
            df=self.diff_rep[order%5]
            w=self.coef(in_rep_.rep_e,out_rep_.rep_e,df)
        # w=w.to(torch.float64)
        w=solve(w).transpose(0,1)
        # w=w.to(torch.float32)
        # print(torch.matrix_rank(w))
        dim_in_rep=in_rep_.dim
        dim_out_rep=out_rep_.dim
        n=w.size(0)
        w=w.reshape(n,dim_out_rep,dim_in_rep,order%5+1)
        self.bases[order][(in_rep,out_rep)]=torch.einsum('ijkl,lmn->ijkmn',w,self.dif_ker[str(order)])
        # qr decomposation to make the basis orthogonal with each other.
        shape=self.bases[order][(in_rep,out_rep)].shape
        self.bases[order][(in_rep,out_rep)],_=torch.qr(self.bases[order][(in_rep,out_rep)].reshape(shape[0],-1).transpose(0,1))
        self.bases[order][(in_rep,out_rep)]=self.bases[order][(in_rep,out_rep)].transpose(0,1).reshape(shape)
        return self.bases[order][(in_rep, out_rep)]



    def base3x3(self, order, in_rep, out_rep):
        '''
            order: an integer representing the order of differential operator.
            in_rep: a string indicate the representation type of input feature
            out_rep: a string indicate the representation type of output feature
            return the bases of the interwiners. of shape (bases x dim_out_rep x dim_in_rep x 3 x 3) comparing to the 5x5 case, the parameters
            is less.
        '''
        if(self.bases[order][(in_rep,out_rep)] is not None):
            return self.bases[order][(in_rep,out_rep)]
        in_rep_=self.rep[in_rep]
        out_rep_=self.rep[out_rep]
        if(self.flip==True):
            df1,df2=self.diff_rep[order%5]
            w1=self.coef(in_rep_.rep_e,out_rep_.rep_e,df1)
            w2=self.coef(in_rep_.rep_m,out_rep_.rep_m,df2)
            w=torch.cat((w1,w2))
        else:
            df=self.diff_rep[order%5]
            w=self.coef(in_rep_.rep_e,out_rep_.rep_e,df)
        # w=w.to(torch.float64)
        w=solve(w).transpose(0,1)
        # w=w.to(torch.float32)
        # print(torch.matrix_rank(w))
        dim_in_rep=in_rep_.dim
        dim_out_rep=out_rep_.dim
        n=w.size(0)
        w=w.reshape(n,dim_out_rep,dim_in_rep,order%5+1)
        self.bases[order][(in_rep,out_rep)]=torch.einsum('ijkl,la,amn->ijkmn',w,self.filter[str(order)],self.dif_ker[str(order)])
        # qr decomposation to make the basis orthogonal with each other.
        b,c1,c,w,h=self.bases[order][(in_rep,out_rep)].shape
        self.bases[order][(in_rep,out_rep)]=compute_base(self.bases[order][(in_rep,out_rep)].reshape(b,-1))
        b=self.bases[order][(in_rep,out_rep)].size(0)
        self.bases[order][(in_rep,out_rep)]=self.bases[order][(in_rep,out_rep)].reshape(b,c1,c,w,h)
        self.bases[order][(in_rep,out_rep)]=self.bases[order][(in_rep,out_rep)][::,::,::,1:4,1:4]
        return self.bases[order][(in_rep, out_rep)]
    
    
    
    
    
    def conv(self, orderlist, in_type, out_type, stride=1, scale=False,group=None):
        '''
            in_type: a list indicate the type of input feature, of the form [in_rep, dim_in]
            out_type: a list indicate the type of output feature, of the form [out_rep, dim_out]
            orderlist: a list containing the order of differential operator
            in_rep: a string indicate the input feature type
            out_rep: a string indicate the output feature type
            dim_in: number of in_rep in the input
            dim_out: number of _out_rep in the output
            scale: if scale = True, then the network is scale equivariant
            group: an arguement in the GroupNorm
            return a convolutiaon layer with input channels as dim_in_rep*dim_in and output channels as dim_out_rep*dim_out
        '''
        
        in_rep,dim_in=in_type
        out_rep,dim_out=out_type
        return conv(self, orderlist , in_rep, out_rep, dim_in, dim_out, stride, scale, group)

    
    
    def conv_fast(self, in_type, out_type, stride=1):
        '''
            in_type: a list indicate the type of input feature, of the form [in_rep, dim_in]
            out_type: a list indicate the type of output feature, of the form [out_rep, dim_out]
            return: a 5x5 equivariant conv layer (conbination of all 0-4 order differential operator)
        '''        
        in_rep, num_in=in_type
        out_rep, num_out=out_type
        if (self.fast_base[(in_rep,out_rep)] is None):
            base=[]
            for i in range(3,8):
                base.append(self.base(i,in_rep,out_rep))
            base=torch.cat(base)
            shape=base.shape
            base,_=torch.qr(base.reshape(shape[0],-1).transpose(0,1))
            base=base.transpose(0,1).reshape(shape)
            self.fast_base[(in_rep,out_rep)]=base
        else:
            base=self.fast_base[(in_rep,out_rep)]
        return conv_fast(base, num_in, num_out, stride)


    
    def conv_fast3x3(self, in_type, out_ype, stride=1):
        '''
            in_type: a list indicate the type of input feature, of the form [in_rep, dim_in]
            out_type: a list indicate the type of output feature, of the form [out_rep, dim_out]
            return: a 3x3 equivariant conv layer (conbination of some 0-4 order differential operator)
        '''
        in_rep, num_in=in_type
        out_rep, num_out=out_type
        if (self.base_3x3[(in_rep,out_rep)] is None):
            base=[]
            for i in range(3,8):
                base.append(self.base3x3(i,in_rep,out_rep))
            base=torch.cat(base)
            shape=base.shape
            base,_=torch.qr(base.reshape(shape[0],-1).transpose(0,1))
            base=base.transpose(0,1).reshape(shape)
            self.base_3x3[(in_rep,out_rep)]=base
        else:
            base=self.base_3x3[(in_rep,out_rep)]
        return conv_fast(base, num_in, num_out, stride)

    # def conv_fast(self, in_type, out_type, stride=1):
    #     in_rep, num_in=in_type
    #     out_rep, num_out=out_type
    #     return nn.Conv2d(num_in*self.rep[in_rep].dim, num_out*self.rep[out_rep].dim,3,stride=stride, padding=1, bias=False)


    def norm(self, in_type, affine=True, momentum=0.1, track_running_stats=True):
        '''
            num_rep: number of representation in the input
            rep: a string indicate the type of representation
            other argument is the same with the standard BatchNorm
        '''
        rep, num_rep=in_type
        dim_rep=self.rep[rep].dim
        return GroupBatchNorm(num_rep, dim_rep, affine, momentum, track_running_stats)
    

    def cuda(self, device=None):
        if(device is None):
            for i in range(5):
                for key,value in self.bases[i].items():
                    if(value is not None):
                        self.bases[i][key]=value.cuda()
        else:
            for i in range(5):
                for key,value in self.bases[i].items():
                    if(value is not None):
                        self.bases[i][key]=value.cuda(device)


    def Restrictlayer(self, in_type):
        _,num_in_rep=in_type
        return GroupRestrict(self.n ,num_in_rep, self.flip)


    def RestrictGroup(self, order=1):
        '''
        order: an integer indicate how many restrict it do, group element decrease to 1/2 every time.
        return: Group
        '''
            
        return Group(n/(2**n), self.flip)

    def GroupRotate(self, x):
        g=torch.inverse(self.diff_rep.g_e).to(x.device)
        x=rotate(x,g)
        dim_rep=self.rep['regular'].dim
        num_in=int(x.size(1)/dim_rep)
        m=self.rep['regular'].rep_e
        x=torch.einsum('ij,bkjmn->bkimn', m, x.reshape(x.size(0), num_in, dim_rep, x.size(2), x.size(3)))\
            .reshape(x.shape)
        return x

    # def Rotate(self, x , in_type):
    #     x=rotate()
# a=torch.randn(1,4,2,2)
# b=torch.zeros(1,4,2,2)
# for i in range(4):
#     b[1,(i+1)%4]=
                        




def test():
    a=torch.randn(1,16*n,28,28)
    g=group(8,True)
    for i in range(5):

        net=g.conv_fast('regular','regular',1,1, scale=True)
        print(g.bases[i][('trivial','regular')])
        y=net(a)
        print(y.shape)
# n=10000
# k=2

# x=torch.randn(1,n,10,10)
# net=nn.Conv2d(n,k*2,5,padding=2)
# nn.init.kaiming_normal_(net.weight)
# y=net(x)**2
# print(torch.sum(y)/(y.size(0)*y.size(1)*y.size(2)*y.size(3)))
# # print(torch.sum(net.weight**2))
# print('ok')
# g=Group(k,True)
# net1=g.conv_fast(['regular',int(n/(2*k))],['regular',1])
# y1=(net1(x))**2
# print(torch.sum(y1)/(y1.size(0)*y1.size(1)*y1.size(2)*y1.size(3)))
# print(torch.sum(net1.base**2))

# y=net1(x)
# print(y)


# test()
# x=torch.randn(1,1,2,2)
# g=Group(8, True)
# a=torch.randn()

# x=torch.arange(8*4).reshape(1,8,2,2).to(torch.float)
# print(x)
# g=Group(4)
# net=g.norm(['regular',2])
# net.train()
# x1=g.GroupRotate(x)
# print(x1)
# y=net(x)
# y1=net(x1)
# y=g.GroupRotate(y)
# print(y)
# print(y1)

# x=torch.randn(1, 4, 2,2)
# g=Group(4)
# x1=g.GroupRotate(x)
# print(x)
# print(x1)
# in_type=['regular',1]
# out_type=['regular',1]
# net=g.conv_fast3x3(in_type,out_type)
# y=net(x)
# y=g.GroupRotate(y)
# y1=net(x1)
# print(y)
# print(y1)





# x=torch.arange(16*4).reshape(1,16,2,2).to(torch.float)
# a=torch.nn.BatchNorm2d(16)
# y=a(x)
# print(y)

# a=rotating(90)
# x_1=rotate(x,a)
# # # device=torch.device('cuda:8')
# g=Group(4,False)
# # net=g.conv_fast(['trivial',1],['regular',1])
# net=g.conv_fast(['trivial',1],['regular',1])
# strict=g.Restrictlayer(['regular',1])
# y=rotate(net(x),a)
# y_1=net(x_1)
# print(y[0])
# print(y_1[0])
def test_time():
    x=torch.randn(64,320,28,28)
    g=Group(8,True)
    net1=g.conv_fast(['regular',20],['regular',20])
    net2=g.conv([0,1,2,3,4,5],['regular',20],['regular',20])
    t1=time.time()
    net1(x)
    t2=time.time()
    net2(x)
    t3=time.time()
    print(t2-t1)
    print(t3-t2)
# test_time()
# print(x)
# print(strict(x))
# print('y',y[0,0])
# print('y_1',y_1[0,1])
# print(a)

# print(g.bases[3][('trivial','regular')])
# g.cuda()
# net=g.conv([0,1,2,3,4],'regular','regular',1,1, scale=True)
# y=net(a)
# print(y.shape)
# f.eval()
# print(f.gn.training)
        
# class A(nn.Module):
#     def __init__(self):
#         super(A, self).__init__()
#         self.tensor=torch.randn(10,10)
#         self.conv=torch.nn.Conv2d(2,2,1)
#     def forward(self, x):
#         return x
# a=A()
# a.cuda()
# print(a.tensor.device)
# x=torch.randn(1,1,28,28)
# a=rotating(45)
# x_1=rotate(x,a)              
# g=group(8,True)
# net=g.conv([],'trivial','trivial',1,1, scale=True)
# y=net(x)
# print(y.shape)
# y_1=net(x_1)
# print(rotate(y[0,0].reshape(1,1,y.size(2),y.size(2)),a))
# print(y_1[0,1])
# print(torch.sum((y_1[0,1,13:17,13:17]-rotate(y[0,0,13:17,13:17].reshape(1,1,4,4),a))**2))




# a=group(8,True)
# d=d_regular(8)
# t1=time.time()
# e=a.base(3,d,d)
# t2=time.time()

# print(e.shape)
# print('Time is {}'.format(t2-t1))




# a=torch.randn(1000,500)
# b=torch.randn(500,800)
# c=torch.matmul(a,b)
# p=solve(c)
# print(torch.sum((torch.matmul(c,p))**2))
# p=torch.pinverse(c)
# g=torch.eye(c.size(1))-torch.matmul(p,c)
# print(torch.matmul(c,g))


# d=solve(c)
# print(torch.sum((torch.matmul(c,d))**2))
# print(torch.matrix_rank(c))

# a=np.random.randn(10,5)
# b=np.random.randn(5,8)


# d=solve_(c)
# print(np.sum((np.matmul(c,d)**2)))
# print(la.matrix_rank(d))  

        



# f=diff_rep(4,True)
# g=f.g_e
# a=torch.tensor([[g[0,0]**2,g[0,0]*g[0,1],g[0,1]**2],[2*g[0,0]*g[1,0],g[0,0]*g[1,1]+g[1,0]*g[0,1],2*g[0,1]*g[1,1]],[g[1,0]**2,g[1,0]*g[1,1],g[1,1]**2]])
# a1,b1=f[2]
# g=f.g_m
# b=torch.tensor([[g[0,0]**2,g[0,0]*g[0,1],g[0,1]**2],[2*g[0,0]*g[1,0],g[0,0]*g[1,1]+g[1,0]*g[0,1],2*g[0,1]*g[1,1]],[g[1,0]**2,g[1,0]*g[1,1],g[1,1]**2]])
# print(a1-a)
# print(b1-b)
# print(b)
# print(a)





            




