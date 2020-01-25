import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

class Net(nn.Module):
    def __init__(self,num_Hidden):
        super(Net, self).__init__()
        self.num_Hidden = num_Hidden
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, num_Hidden)
        self.fc2 = nn.Linear(num_Hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x



**Training Function**

num_Hidden = 200
train_input, train_target = Variable(train_input), Variable(train_target)

model, criterion = Net(num_Hidden), nn.MSELoss()
eta, mini_batch_size = 1e-1, 100

def train_model(model,train_input, train_target, mini_batch_size):
    for e in range(0, 25):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        #print(e, sum_loss)

train_model(model,train_input, train_target, mini_batch_size)

**Test error**

def compute_nb_errors(model,input,target,mini_batch_size):
    nb_errors = 0
    for b in range(0, train_input.size(0), mini_batch_size):
            output = model(input.narrow(0, b, mini_batch_size))
            _,prediction_class = output.max(1)
            for c in range( mini_batch_size):
                if target[b+c, prediction_class[c]] <= 0:
                    nb_errors += 1
                    
    return   nb_errors
                
            
    

compute_nb_errors(model,train_input,train_target,mini_batch_size)

for i in range(0,10):
    train_model(model,train_input, train_target, mini_batch_size)
    print(compute_nb_errors(model,train_input,train_target,mini_batch_size))
    
        
    

**Influence of the number of Hidden layers**

for Number_Hidden_Layers in [ 10, 50, 200, 500, 1000 ]:
    model = Net(Number_Hidden_Layers)
    train_model(model, train_input, train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net Number_Hidden_Layers={:d} {:0.2f}%% {:d}/{:d}'.format(Number_Hidden_Layers,
                                                              (100 * nb_test_errors) / test_input.size(0),
                                                              nb_test_errors, test_input.size(0)))

**Three Convolutional Layers**

class Net2(nn.Module):           ## 4 Layer Convolution
    def __init__(self, Number_Hidden_Layers):
        super(Net2, self).__init__()
        self.Number_Hidden_Layers = Number_Hidden_Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128*1*1, Number_Hidden_Layers)
        self.fc2 = nn.Linear(Number_Hidden_Layers, 10)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 128*1*1)  ### Flattening
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

for Number_Hidden_Layers in [ 10, 50, 200, 500, 1000 ]:
    model = Net2(Number_Hidden_Layers)
    train_model(model, train_input, train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net Number_Hidden_Layers={:d} {:0.2f}%% {:d}/{:d}'.format(Number_Hidden_Layers,
                                                              (100 * nb_test_errors) / test_input.size(0),
                                                              nb_test_errors, test_input.size(0)))



