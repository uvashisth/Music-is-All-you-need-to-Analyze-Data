import numpy as np
import os
import logging
import configparser
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer


# to import parent-level modules
os.chdir('Pitch-Based Approach')
sys.path.append('..')


# import local modules
from preprocessing.preprocessing import PreprocessingTrainingData
from model.StackedLSTM import StackedLSTM,init_weights
from model.StackedLSTM import StackedLSTM, init_weights
from utils.split_train_val import split_train_val

# check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# read the configuration file
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


# get data from preprocessing.py
network_input,network_output,max_midi_number,min_midi_number,int_to_note = PreprocessingTrainingData().preprocess_notes(train_dir)
network_input, network_output = network_input.to(device), network_output.to(device)


'''
update the config file, this information is required when predicting
'''
config.set('Save', 'initial_seq', str([network_input[0][1:].cpu().numpy().tolist()]))
config.set('Save', 'int2note', str(int_to_note))
config.set('Save', 'max_note', str(max_midi_number))
config.set('Save', 'min_note',str(min_midi_number ))

with open('config.ini', 'w') as cfg:
    config.write(cfg)



# divide the dataset into train/val
train_loader, val_loader = split_train_val(train_size, network_input, network_output, batch_size)


model = StackedLSTM(input_size,hidden_size,num_layer,output_size,batch_size)
model.apply(init_weights)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optimizer.AdamW(model.parameters())

# make sure to transfer model to GPU(if available) after initializing the optimizer
model.to(device)


min_val_loss = np.Inf
for e in range(epochs):
    '''

    '''
    train_loss = 0
    val_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    
    for inputs,labels in train_loader:
        
        # zero accumulated gradients
        model.zero_grad()
       
        # get the output from the model
        output = model.forward(inputs)
    
        # calculate the loss and perform backprop
        loss = criterion(output,labels.long())
        
        loss.backward()
        
        # calculate training accuracy
        output = F.softmax(output, dim = 1)
        top_p, top_class = output.topk(1, dim=1)
        equals = top_class == labels.long().view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        train_loss += loss.item()
              
    # set the model in evaluation mode
    model.eval()
    for inputs, labels in val_loader:
        '''
        

        '''        
        
        output = model.forward(inputs)
       
        loss = criterion(output,labels.long())
        val_loss += loss.item()
        
        # calculate validation accuracy
        output = F.softmax(output, dim = 1)
        top_p, top_class = output.topk(1, dim=1)
        equals = top_class == labels.long().view(*top_class.shape)
        val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    # set the model in training mode
    model.train()
    
    # Averaging losses
    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    val_accuracy = val_accuracy/len(val_loader)
    train_accuracy = train_accuracy/len(train_loader)
    
    
    print('Epoch: {}\tTrain Loss: {:.7f} \tVal Loss:{:.7f} \tTrain Acc: {:.7}% \tVal Acc: {:.7f}%'.format(e, train_loss, val_loss, train_accuracy*100,val_accuracy*100))
    
    # saving the model if validation loss is decreased
    if val_loss <= min_val_loss:
        print('Validation Loss decreased from {:6f} to {:6f}, saving the model weights'.format(min_val_loss, val_loss))
        torch.save(model.state_dict(), '{}'.format(weights_loc))
        min_val_loss = val_loss




