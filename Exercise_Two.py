import torch
import math

**Activation Fuction

def sigma(x):
    return torch.tanh(x)

def dsigma(x):
    return  1 - (sigma(x)**2)

**Loss**

def loss(v,t):
    return  torch.pow(torch.norm(v-t),2)

def dloss(v,t):
    return 2*(v-t)
    

from dlc_practical_prologue import *

Data = load_data()

train_data, train_labels, test_data,test_labels = load_data(one_hot_labels=True, 
                                                                        normalize=True)

zeta = 0.9
train_data = train_data * zeta
test_labels = test_labels * zeta
eta = 1e-1/ train_data.size(0)

hidden = 50
input_size = 784
output_size = 10


nb_classes = train_labels.size(1)
nb_train_samples= train_data.size(0)


w1 =  torch.empty(hidden,input_size ).normal_(mean=0, std=1e-6)
#x1 =  torch.empty(input_size,1).normal_(mean=0, std=1e-6)
b1 = torch.empty(hidden).normal_(mean=0, std=1e-6)

w2=  torch.empty(output_size,hidden ).normal_(mean=0, std=1e-6)
#x2 =  torch.empty(hidden,1).normal_(mean=0, std=1e-6)
b2 = torch.empty(output_size).normal_(mean=0, std=1e-6)

dl_dw1 = torch.empty(w1.size())
dl_dw2 = torch.empty(w2.size())
dl_db1 = torch.empty(b1.size())
dl_db2 = torch.empty(b2.size())

def forward_pass(w1,b1,w2,b2,x):
    x0 = x
    s1 = torch.mv(w1,x0)+ b1
    x1 = sigma(s1)
    s2 = torch.mv(w2,x1.T) + b2
    x2 = sigma(s2)
    return x0,s1,s2,x1, x2


def backward_pass(w1,b1,w2,b2,t,X,s1,x1,s2,x2,dl_dw1,dl_db1,dl_dw2,dl_db2):
    first = dloss(x2,t)
    dl_ds2 = first * dsigma(s2)
    dl_dw2=torch.mm(dl_ds2.view(-1,1), x1.view(1,-1))
    dl_db2=dl_ds2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dl_dx1*dsigma(s1)
    dl_db1=dl_ds1
    dl_dw1=torch.mm(dl_ds1.view(-1,1), X.view(1,-1))
    return dl_db1, dl_db2,dl_dw1,dl_dw2


for k in range(1000):

    # Back-prop

    acc_loss = 0
    nb_train_errors = 0

    
    dl_db1.zero_()
    dl_db2.zero_()
    dl_dw1.zero_()
    dl_dw2.zero_()

    for n in range(train_data.size(0)):
        x0,s1,s2,x1, x2 = forward_pass(w1, b1, w2, b2, train_data[n])

        pred = x2.argmax()
        if train_labels[n, pred] < 0.5: nb_train_errors = nb_train_errors + 1
        acc_loss = acc_loss + loss(x2, train_labels[n])

        k1,k2,k3,k4 = backward_pass(w1, b1, w2, b2,
                      train_labels[n],
                      x0, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2)
        
        dl_db1=dl_db1+ k1
        dl_db2 = dl_db2 + k2
        dl_dw1 = dl_dw1+k3
        dl_dw2 =dl_dw2+ k4

#     # Gradient step

    w1 = w1 - eta * dl_dw1
    b1 = b1 - eta * dl_db1
    w2 = w2 - eta * dl_dw2
    b2 = b2 - eta * dl_db2

     # Test error

    nb_test_errors = 0

    for n in range(test_data.size(0)):
        _,_,_,_, x2= forward_pass(w1, b1, w2, b2, test_data[n])

        pred = x2.argmax()
        if test_labels[n, pred] < 0.5: nb_test_errors = nb_test_errors + 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_data.size(0),
                  (100 * nb_test_errors) / test_data.size(0)))



