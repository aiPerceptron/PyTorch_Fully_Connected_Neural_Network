# Summary

I created a neural network with one of the most popular machine learning packages, PyTorch.

# Introduction

  I have three goals when I make this model. My first goal is to remove as many dependencies on other packages as possible, and to rely as much as possible on PyTorch. My second goal is to add batching to my model's training and testing. Batching is the ability to split up a dataloader into small chunks (like 5 or 10 pieces of data per loop). This allows the model to have more performance, as you aren't feeding it the entire dataset at once. 
  
  The Tensor is the simplest array structure in PyTorch. A dataset is a special array in PyTorch, and a dataloader allows you to batch datalists.
  
  My third and final goal is to expand on my knowledge on advanced machine learning scoring. This will allow me to adjust and improve the machine learning model better. If my model were a student taking a school test, I'd be the teacher learning how to give the student higher quality feedback.

# Methodology

  For my first goal, I want to replace my dependency on Scikitlearn's train_test_split function with PyTorch's random_split function. In order to do this though, I have to convert my arrays to datasets. 
```
ds_Xy = TensorDataset(X,y)
```
Now I split them up.
```
ds_train, ds_test = random_split(ds_Xy, [0.8,0.2])
```

For my second goal, I need to convert my datasets to dataloaders.
```
dl_train = DataLoader(ds_train,batch_size=batch_size,shuffle=True)
dl_test = DataLoader(ds_test,batch_size=batch_size)
```
Now instead of loading the whole dataset into my models training loop, I can just slowly load in batches of data. 
```
for xb,yb in dl_train:
```

I will now walk you through all the steps in the comments in the training loop (which uses all PyTorch).

```
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
```

  Now for my last goal, I need to learn a few new methods to score my model. There are 6 methods, and each one measures a slightly different thing. I won't go into the details here because I made a seperate document for this.

  The accuracy score is just how many answers the model got right.

  The precision score is how *precise* your model is. If you want to make your model get a true positive as many times as posible use the precision score. (true positve means when a model guesses something is positive (like you test positive for a sickness) and it's right). 
    
  Recall score tests for false negatives. People like doctors would want to use this, because let's say someone comes in for a cancer screening and the AI model they use to test people with gives a false negative. Well then the person thinks they don't have cancer when they really do and it's worsening unnoticed. This is why you would want to use recall score. 
  
  F Score is just a combination of Preciison score and Recall score.

  Average score is just the average of all of the scores.

  The confusion matix display displays true positives false positives true negatives and false negatives.

I was successfully able to measure all the scores.

```
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
#set y_cat to y_cat.cpu().numpy()
acc_score = accuracy_score(y_concat.cpu().numpy(), y_hat_concat.cpu().numpy())
#print(acc_score)

p_score = precision_score(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy(),average="macro")

#print(p_score)

r_score = recall_score(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy(),average="macro")

#print(r_score)

f_score = f1_score(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy(),average="macro")

#print(f_score)

average_score = (acc_score + p_score + r_score + f_score)/4*100
print(average_score)

c_matrix = confusion_matrix(y_cat.cpu().numpy(), y_hat_cat.cpu().numpy())

disp = ConfusionMatrixDisplay(c_matrix)
disp.plot()
c_matrix
```
Evem though I'm using Scikitlearn for the analytics, that's fine as long as im not using it for training.

# Results
  I was able to successfully build a machine learning model using mostly PyTorch. I was also able to train the model using dataset batching. The only things that I'm using Scikitlearn for now are `make_classification` which allwos me to generate a random dataset, and scoring after the training. I'd say this was a success.
# Conclusions & Future Works
If you see `.to(device)` anywhere in my code such as here, `xb,yb = xb.to(device),yb.to(device)` it is because I am checking for an Nvidia GPU, and if one is there, it will send the training data to the gpu. This will make the training process way faster.

My code for the GPU check
```
isgpu = torch.cuda.is_available()
if isgpu is True:
    device = torch.device("cuda")
    print("GPU is available")
else:
    print("GPU isn't available")
    device = torch.device("cpu")
```


