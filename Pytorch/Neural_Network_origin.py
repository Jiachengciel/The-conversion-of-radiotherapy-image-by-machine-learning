"""
Created on 2019-05-20
Updated on 2019-06-06
Company: DOSIsoft
Author: Jiacheng
"""

# =================Library==============================
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from Pytorch_Input_data import *


# =================Neural Network=======================
INPUT_NODE = 7
HIDDEN_LAYER1_NODE = 30
HIDDEN_LAYER2_NODE = 5
HIDDEN_LAYER3_NODE = 1
HIDDEN_LAYER4_NODE = 1
OUTPUT_NODE = 1

# =================Parameters===========================
LEARNING_RATE = [0.5, 0.05, 0.005, 0.0005, 0.00005]
# Learning_Rate = 0.0001
Learning_Rate = 0.00005
LAMBDA = 0.0001
Batch_size = 128
Epoch = 80
Epoch_base = 0

# =================Store path===========================
PATH_MODEL = './model/net_params.pkl'
PARAMETERS = 'lr'+str(Learning_Rate) + '_Epoch'+str(Epoch) + '_lambda'+str(LAMBDA)
PATH_PARAMETERS = './model_origin/checkpoint_' + PARAMETERS + '.pth.tar'

# PATH_MODEL = './model/net_params.pkl'
# PATH_PARAMETERS = './model_origin/checkpoint_lamda0.001_SGD_lr0.00005.pth.tar'
# PATH_PARAMETERS = './model_origin/checkpoint1_lamda0.001_SGD_lr0.00005.pth.tar'


# =================Class of Neural Network==============
class Neural_Network(nn.Module):

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(Neural_Network, self).__init__()

        self.ANN = nn.Sequential(
            # 1
            nn.Linear(input_dim, hidden1_dim),
            nn.Tanh(),
            # 2
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.Tanh(),
            # 3
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.Sigmoid(),
        )

        # Linear function for increasing value: 1 --> 1
        self.out = nn.Linear(hidden3_dim, output_dim)

    def forward(self, X):
        y = self.ANN(X)

        #  Increasing the value
        out = self.out(y)

        return out


# =================Data Processing========================
# Get all the data
X_np, Y_np, _, _ = get_train_data_origin()

# Split the train and the validaton set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_np, Y_np, test_size=0.1, random_state=2)

print("X Max: %f, Min: %f, Y Max: %f, Min: %f" % (X_np.max(), X_np.min(), Y_np.max(), Y_np.min()))
print("\nQuantity of train examples: {}\nX_train shape: {} Y_train shape: {}".format(X_train.shape[0], X_train.shape,
                                                                                   Y_train.shape))
print("Quantity of validation examples: {}\nX_val shape: {} Y_val shape: {}".format(X_val.shape[0], X_val.shape,
                                                                                    Y_val.shape))
print('\nLearning rate: ', Learning_Rate)
print('Weight Decay: ', LAMBDA)


# =================Tensor Processing========================
# Creat tensor
X_Train = torch.from_numpy(X_train)
Y_Train = torch.from_numpy(Y_train).type(torch.FloatTensor).view(-1, 1)

X_Test = torch.from_numpy(X_val)
Y_Test = torch.from_numpy(Y_val).type(torch.FloatTensor).view(-1, 1)

# load data into dataset
train = torch.utils.data.TensorDataset(X_Train, Y_Train)
test = torch.utils.data.TensorDataset(X_Test, Y_Test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size=Batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=Batch_size, shuffle=False)


# =================Model Environment========================
# Creat Neural Network
model = Neural_Network(INPUT_NODE, HIDDEN_LAYER1_NODE, HIDDEN_LAYER2_NODE, HIDDEN_LAYER3_NODE, OUTPUT_NODE)

# Loss function
MSE = nn.MSELoss(reduction='mean')

# Moves all model parameters and buffers to the GPU
if torch.cuda.is_available():
    print("\nGPU Process!!!\n")
    model.cuda()
    MSE.cuda()

# Optimizer
# Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate, weight_decay=0.03)

# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate, momentum=0.9, weight_decay=LAMBDA)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

print("Neural Network structure:\n", model)

# Check whether there is a model
IsContinue = os.path.exists(PATH_PARAMETERS)
if IsContinue:
    print("Model exists, continue training!!!")
    checkpoint = torch.load(PATH_PARAMETERS)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    Epoch_base = checkpoint['epoch']
    scheduler = scheduler.load_state_dict(checkpoint['scheduler'])

else:
    print("No model, begin training!!!")


# ==================ANN model training=======================
count = 0
loss_list = []
loss_list_temp = []
loss_val_list = []

iteration_list = []
accuracy_list = []
r2_list = []

for epoch in range(Epoch_base, Epoch):
    for step, (input, dose) in enumerate(train_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            dose = dose.cuda()

        Input = Variable(input)
        Dose_TPS = Variable(dose)

        # clear gradients
        if not IsContinue:
            optimizer.zero_grad()

        # Forward propagation
        dose_previson = model(Input.float())

        # Calculate the loss
        loss = MSE(dose_previson, Dose_TPS)

        # Backward propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

        # print("Epoch: {} Iteration: {}, Loss: {}".format(epoch + 1, count, round(loss.data.item(), 2)))

        if count % 4000 == 0:

            loss_list.append(loss.data)
            iteration_list.append(count)

            # store the checkpoint
            # torch.save(model.state_dict(), PATH_MODEL)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss
            }, PATH_PARAMETERS)

            loss_list_temp.clear()
            r2_list.clear()
            for i, (inputs, dose) in enumerate(test_loader):
                # Calculate accuracy
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    dose = dose.cuda()

                test = Variable(inputs)
                Dose_Test = Variable(dose)

                dose_prevision_test = model(test.float())
                dose_prevision_test_np = dose_prevision_test.cpu().detach()

                loss_val = MSE(dose_prevision_test, Dose_Test)
                r2 = r2_score(Dose_Test.cpu().detach().numpy(), dose_prevision_test_np) * 100

                # free some space
                del test
                del Dose_Test
                del dose_prevision_test
                del dose_prevision_test_np

                # Store the value
                loss_list_temp.append(loss_val.data.item())
                r2_list.append(r2)

            loss_val = np.mean(loss_list_temp)
            loss_val_list.append(loss_val)
            r2 = np.mean(r2_list)
            accuracy_list.append(r2)

            # if don't decrease, change the learning rate
            scheduler.step(loss_val)

            # print results
            if count % 20000 == 0:
                # torch.save(model.state_dict(), PATH_MODEL)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': loss
                }, PATH_PARAMETERS)

                print("Epoch: {} Iteration: {}, Loss: {}, Loss_val : {}, Accuracy: {} %".format(epoch + 1, count,
                                                                                                round(loss.data.item(),
                                                                                                      6),
                                                                                                round(loss_val, 6),
                                                                                                round(r2, 4)))
                # print("Epoch: {} Iteration: {}, Loss: {}".format(epoch+1, count, round(loss.data.item(), 2)))


# ==================Visualization========================
Path_loss = './Loss_origin/'
plt.plot(iteration_list, loss_list, color="red", label='Training Loss')
plt.plot(iteration_list, loss_val_list, color="blue", label='Cross Validation Loss')
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN")
plt.legend()

plt.savefig(Path_loss + PARAMETERS + '.png')
plt.show()

plt.plot(iteration_list, accuracy_list, color="red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN_Accuracy")
plt.savefig(Path_loss + 'Accuracy_' + PARAMETERS + '.png')
plt.show()
