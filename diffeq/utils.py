import torch
import e2cnn
from e2cnn import gspaces
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
    print(height,width)
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
    height,width=x.shape[-2:]
    cen=torch.tensor([(height-1)/2.,(width-1)/2.])
    y=torch.zeros_like(x)
    
    for i in range(height):
        for j in range(width):
            f=torch.tensor([0.])
            t=torch.tensor([i,j]).to(torch.float32)
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




def scale():
    #TODO



# x=np.arange(25)
# x=x.reshape(1,1,5,5)
# print(x)
# a=np.array([[-1.,0.],[0.,1.]])
# print(numpy_rotate(x,a))





class diff_rep:
    ### n represent the order of the rotation
    def __init__(self,n,flip=False):
        self.rep_e=[]
        self.rep_m=[]
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
        a=self.rep_e[self.order-1]
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
            a=self.rep_m[self.order-2]
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
            return self.rep_e[i-1],self.rep_m[i-1]
        else:
            return self.rep_e[i-1]


class c_regular:
    def __init__(self,n):
        self.type='cn'
        self.n=n
        self.rep_e=torch.zeros(n,n)
        self.rep_e[0:n-1,1:n]=torch.eye(n-1)
        self.rep_e[n-1,0]=1.
        self.dim=n


class d_regular:
    def __init__(self,n):
        self.n=n
        self.type='dn'
        self.rep_e=torch.zeros(2*n,2*n)
        self.rep_e[0:n-1,1:n]=torch.eye(n-1)
        self.rep_e[n-1,0]=1.
        self.rep_e[n+1:2*n,n:2*n-1]=torch.eye(n-1)
        self.rep_e[n,2*n-1]=1.
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
    print(n)
    u,s,v=torch.svd(A,some=True)
    return v[::,n:A.size(1)]



def solve_(A):
    G=la.pinv(a)
    print(G.shape)
    w=np.eye(A.shape[1])-np.matmul(G,A)
    n=la.matrix_rank(w)
    u,s,v=la.svd(w)
    return u[::,0:n]


class iden(nn.Module):
    def __init__(self):
        super(iden,self).__init__()
    
    
    
    def forward(self,x):
        return x




class conv_base(nn.Module):
    def __init__(self, base, dim_in, dim_out, stride=1):
        ''' 
            bases: number of basis kernel in a specified type
            param: parameter tensor of shape dim_in x dim_out x bases
            base: bases of a specified type of kernel, shape of  bases x dim_rep_out x dim_rep_in x kernel_size x kernel_size
        '''
        super(conv_base,self).__init__()
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.dim_rep_in=base.size(2)
        self.dim_rep_out=base.size(1)
        self.base=base
        self.bases=base.size(0)
        self.parma=torch.nn.Parameter(torch.randn(dim_in, dim_out, self.bases))
        self.stride=stride
        self.size=self.base.size(4)
    
    
    def forward(self,x):
        #get the kernel of the conv from the base
        kernel=torch.einsum('ijk,kmnpq->injmpq',self.parma, self.base)\
            .reshape(self.dim_out*self.dim_rep_out,self.dim_in*self.dim_rep_in,self.size,self.size)
        return nn.functional.conv2d(x,kernel,bias=None, stride=self.stride, padding=math.floor(self.size/2))



class conv(nn.Module):
    def __init__(self, baselist, dim_in, dim_out, stride, exist, scale=False,group=None):
        '''
            baselist: a list consists of base in conv_base.
            scale: the net is scale equivariant or not.
            group: is the arguments num_groups in the GroupNorm, it is dim_in as default.
            exist: if exist = True, then the 0 order differential operator is in the conv.
            the other argument is the same with conv_base.
        '''
        super(conv,self).__init__()
        self.dim_in=dim_in
        self.dim_out=dim_out
        list=[]
        for i in baselist:
            list.append(conv_base(i, dim_in, dim_out,  stride))
        if(exist==True):
            list.append(nn.Conv2d(dim_in*i.size(2),dim_out*i.size(1),1, stride=stride,padding=0, bias=False))

        self.convlist=nn.ModuleList(list)
        if (group==None):
            group=dim_out
        if scale==False:
            self.gn=iden()
        else:
            self.gn=torch.nn.GroupNorm(group,i.size(1)*dim_out)
    def forward(self,x):
        y=self.gn(self.convlist[0](x))
        for i in range(1,len(self.convlist)):
            y=y+self.gn(self.convlist[i](x))
        return y





class group:
    def __init__(self, n, flip=False):
        '''
            n: n means the group has n pure rotational element.
            flip: wether the group include flip element
        '''
        self.dif_ker={'1':torch.tensor([[[0.,0.,0.],[-0.5,0.,0.5],[0.,0.,0.]],[[0.,0.5,0.],[0.,0.,0.],[0.,-0.5,0.]]]),\
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
                    [[0.,0.,1.,0.,0.],[0.,0.,-4.,0.,0.],[0.,0.,6.,0.,0.],[0.,0.,-4.,0.,0.],[0.,0.,1.,0.,0.]]])}
        self.n=n
        self.flip=flip
        self.diff_rep=diff_rep(n, flip)
        self.rep={'regular':None, 'trivial':trivial()}
        if(flip==True):
            self.rep['regular']=d_regular(n)
        else:
            self.rep['regular']=c_regular(n)
        self.bases=[{('regular','regular'):None, ('regular','trivial'):None, ('trivial','trivial'):None,('trivial','regular'):None}]*5
    
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
        if(self.bases[order][(in_rep,out_rep)]!=None):
            return self.bases[order][(in_rep,out_rep)]
        in_rep_=self.rep[in_rep]
        out_rep_=self.rep[out_rep]
        if(self.flip==True):
            df1,df2=self.diff_rep[order]
            w1=self.coef(in_rep_.rep_e,out_rep_.rep_e,df1)
            w2=self.coef(in_rep_.rep_m,out_rep_.rep_m,df2)
            w=torch.cat((w1,w2))
        else:
            df=self.diff_rep[order]
            w=self.coef(in_rep_.rep_e,out_rep_.rep_e,df)
        print(torch.matrix_rank(w))
        # w=w.to(torch.float64)
        w=solve(w).transpose(0,1)
        # w=w.to(torch.float32)
        
        dim_in_rep=in_rep_.dim
        dim_out_rep=out_rep_.dim
        n=w.size(0)
        w=w.reshape(n,dim_out_rep,dim_in_rep,order+1)
        self.bases[order][(in_rep,out_rep)]=torch.einsum('ijkl,lmn->ijkmn',w,self.dif_ker[str(order)])
        return self.bases[order][(in_rep, out_rep)]
    
    def conv(self, orderlist, in_rep, out_rep, dim_in, dim_out, stride=1, scale=False,group=None):
        '''
            orderlist: a list containing the order of differential operator
            in_rep: a string indicate the input feature type
            out_rep: a string indicate the output feature type
            dim_in: number of in_rep in the input
            dim_out: number of _out_rep in the output
            scale: if scale = True, then the network is scale equivariant
            group: an arguement in the GroupNorm
            return a convolutiaon layer with input channels as dim_in_rep*dim_in and output channels as dim_out_rep*dim_out
        '''
        exist=False
        if 0 in orderlist:
            exist=True
            orderlist.remove(0)
        base=[]
        for i in orderlist:
            base.append(self.base(i,in_rep, out_rep))
        return conv( base, dim_in, dim_out, stride, exist, scale, group)


        
  
x=torch.randn(1,16,28,28)
a=rotating(45)
x_1=rotate(x,a)              
g=group(8,True)
net=g.conv([0,1,2,3,4],'regular','trivial',10,10, scale=True)
y=net(x)
print(y.shape)
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





            




