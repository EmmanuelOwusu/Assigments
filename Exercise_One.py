import torch
import time 



**QUESTION 1**


#The objective of this session is to practice with basic tensor manipulations in pytorch, to understand
#the relation between a tensor and its underlying storage, and get a sense of the efficiency of tensorbased computation compared to their equivalent python iterative implementations.

A=torch.full((13,13), 1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
A

A[:,[1,6,11]],A[[1,6,11],:]=2,2

A[3:5,3:5], A[8:10,8:10],A[3:5,8:10],A[8:10,3:5]=3,3,3,3


A



**QUESTION 2**

#Generate two square matrices of dimension 5000 × 5000 filled with random Gaussian coefficients,
#compute their product, measure the time it takes, and estimate how many floating point products
#have been executed per second (should be in the billions or tens of billions).

%%time

#B=torch.randn(500,500)

from time import perf_counter

B = torch.empty((500,500)).normal_(mean=0.5, std=1)
C = torch.empty((500,500)).normal_(mean=0.3, std=2)
time1 = perf_counter()
D=torch.mm(B,C)
time2= perf_counter()
t_current =time2-time1
t_current
f"The time to compute the matrix multipilcation is {t_current}"

Floops = ((5000)**3)/t_current
Floops



**QUESTION 3**

#Write a function mul row, using python loops (and not even slicing operators), that gets a 2d tensor
#as argument, and returns a tensor of same size, equal to the one given as argument, with the first
#row kept unchanged, the second multiplied by two, the third by three, etc.

def mul_row(m):
    for i in range(m.size(0)):
        for j in range(m.size(1)):
            m[i,j]= torch.mul(i+1,m[i,j])
    return m
            
            

time1_start = perf_counter()
m = torch.full((4,8),2)
res=mul_row(m)
time1_stop = perf_counter() 
t_current
f"The time used to execute the program is {t_current} seconds"

def mul_row_fast(m):
    for i in range(m.size(0)):
        for j in range(m.size(1)):
            m[i,j]= torch.mul(i+1,m[i,j])
    return m
            


time2_start = perf_counter()
m=torch.full((1000,400),2)
res=mul_row(m)
time2_stop = perf_counter() 
t_current
f"The time used to execute the program is {t_current} seconds"

