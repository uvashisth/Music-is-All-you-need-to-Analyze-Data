import numpy as np
import os
import logging
import configparser
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer


#to import parent-level modules
os.chdir('Temporal-Based Approach')
sys.path.append('..')


from preprocessing.preprocessing import PreprocessingTrainingDataM2
from model.StackedLSTM import StackedLSTM,init_weights
from model.StackedLSTM import StackedLSTM, init_weights
from utils.split_train_val import split_train_val

#check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

batch_size = int(config['Train']['train_batch_size'])
train_size = float(config['Train']['train_size'])
clip = int(config['Train']['clip'])
epochs = int(config['Train']['epochs'])
train_dir =  config['Train']['train_dir']
input_size = int(config['Common']['input_size'])
hidden_size = int(config['Common']['hidden_size'])
num_layer = int(config['Common']['num_layer'])
sequence_length = int(config['Common']['sequence_length'])
output_size = int(config['Common']['output_size'])
weights_loc = config['Common']['weights_loc']

#get data from preprocessing.py
network_input,network_output = PreprocessingTrainingDataM2().preprocess_notes(train_dir)
network_input, network_output = torch.tensor(network_input).to(device), torch.tensor(network_output).to(device)

print(network_input.shape)

#update the config file
config.set('Save', 'initial_seq', str([network_input[0].cpu().numpy().tolist()]))

with open('config.ini', 'w') as cfg:
    config.write(cfg)


print(network_input.shape)

#divide the dataset into train/val
train_loader, val_loader = split_train_val(train_size, network_input, network_output, batch_size)


model = StackedLSTM(input_size,hidden_size,num_layer,output_size,batch_size)
model.apply(init_weights)

cat_criterion = nn.BCEWithLogitsLoss(reduce = True).to(device)
note_criterion = nn.MSELoss().to(device)
vel_criterion = nn.MSELoss().to(device)

optimizer = optimizer.AdamW(model.parameters())

#make sure to transfer model to GPU after initializing optimizer
model.to(device)

min_val_loss = np.Inf
for e in range(epochs):
    
    train_loss = 0
    val_loss = 0
    
    for inputs,labels in train_loader:
        '''
        Creating new variables for the hidden state, otherwise
        we'd backprop through the entire training history
        '''
        
        # zero accumulated gradients
        model.zero_grad()
       
        # get the output from the model
        output = model.forward(inputs)
    
        # calculate the loss and perform backprop
        cat_loss = cat_criterion(output[:,1:98],labels[:,1:98])
        reg_loss = note_criterion(output[:,0],labels[:,0])
        vel_loss = vel_criterion(output[:,-1],labels[:,-1])

        loss = cat_loss + reg_loss + vel_loss

        loss.backward()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        train_loss += loss.item()
              
    model.eval()
    for inputs, labels in val_loader:
                
        output = model.forward(inputs)
       
        cat_loss = cat_criterion(output[:,1:98],labels[:,1:98])
        reg_loss = note_criterion(output[:,0],labels[:,0])
        vel_loss = vel_criterion(output[:,-1],labels[:,-1])

        loss = cat_loss + reg_loss + vel_loss

        val_loss += loss.item()
        
    model.train()
    
    #Averaging losses
    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    
    
    print('Epoch: {}\tTrain Loss: {:.7f} \tVal Loss:{:.7f}'.format(e, train_loss, val_loss))
    
    #saving the model if validation loss is decreased
    if val_loss <= min_val_loss:
        print('Validation Loss decreased from {:6f} to {:6f}, saving the model weights'.format(min_val_loss, val_loss))
        torch.save(model.state_dict(), '{}'.format(weights_loc))
        min_val_loss = val_loss




