# Fully Connected Neural Network (My FIRST Neural Network in PyTorch)

#1: creating the data
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset,DataLoader,random_split

# Inspecting the hardware # preparation 1

isgpu = torch.cuda.is_available()
if isgpu is True:
    device = torch.device("cuda")
    print("GPU is available")
else:
    print("GPU isn't available")
    device = torch.device("cpu")

# Hyperparameters # preparation 2

n_epochs = 10
lr = 0.01
n_Neurons1 = 4

# Getting the data #1

# Random Data Set
N = 1000 # how many people(or samples) we have, or the rows
F = 2 # the features, or the columns

n_classes = 3
batch_size = 5
X, y = make_classification(n_samples=N, # relates to both X(features) and y (targets/class labels)
                           n_features=F, # number of different features (saved to X) but it's read by y
                           n_informative=2, # adds on to X(features)
                           n_redundant=0, # adds on to X(features)
                           n_classes=n_classes, # created because of the X(features) (saved to y(targets/class labels))
                           n_clusters_per_class = 1, # how many clusters are in 1 class (related to X(features))
                           class_sep=1) # how far apart each class is from each other (related to X(features))
X = torch.tensor(X).to(torch.float32)
y = torch.tensor(y).to(torch.int64) # !!! y needs to be whole numbers for CrossEntropyLoss. it needs to be int64 or long.
ds_Xy = TensorDataset(X,y)

ds_train, ds_test = random_split(ds_Xy, [0.8,0.2])

dl_train = DataLoader(ds_train,batch_size=batch_size,shuffle=True)
dl_test = DataLoader(ds_test,batch_size=batch_size)

# Plotting the data
plt.scatter(X[:,0],X[:,1],c=y)

# Picking the model #2

model = nn.Sequential(nn.Linear(F,n_Neurons1), # the linear part
                      nn.Sigmoid(), # the activation part
                      nn.Linear(n_Neurons1,n_classes)) # you dont need the activation part for the last neuron, since we will do it after
model = model.to(device)
# In scikitlearn you create this model by using this line of code
# model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(n_Neurons1,))
# the comma is there because you need it to create tuple that has only a single item

# Training #3

model.train() # prepares the model for training
loss_function = nn.CrossEntropyLoss() # object for finding the loss
optimizer = optim.SGD(model.parameters(), lr=lr) # object for adjusting the machine learning models weights
for epoch in tqdm(range(n_epochs)): # loop to loop over the loop for a long time
    for xb,yb in dl_train: # loops over the batches
        xb,yb = xb.to(device),yb.to(device) # moves the data to the GPU
        ml_out = model(xb) # the models answer
        loss = loss_function(ml_out,yb) # you're comparing ml_out with y[i] here to see the loss (aka what adjustments you should make next time)
        optimizer.zero_grad() # sets gradient to 0
        loss.backward() # compute the gradient
        optimizer.step() # goes down the gradient in little baby steps (to make the loss less)

# Calculating the percent accuracy
#y_hat = ml_out.argmax(axis=1)
#compare_y_test_y_hat = y_hat == y_test

#percent_accuracy = (sum(compare_y_test_y_hat)/len(y_hat))*100
#print(str(int(percent_accuracy))+"%")
