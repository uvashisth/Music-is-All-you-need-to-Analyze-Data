import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


# divide the dataset into train/val
train_loader, val_loader = split_train_val(train_size, network_input, network_output, batch_size)


#load the weights of the LSTM model
model = StackedLSTM(input_size,hidden_size,num_layer,output_size, batch_size)
model.load_state_dict(torch.load('{}'.format(weights_loc)))

#set the model in evaluation mode
model.eval()
model.to(device)

actual_list = []
predictd_list = []


for inputs, labels in val_loader:
    output = model.forward(inputs)
            
    # calculate validation accuracy
    output = F.softmax(output, dim = 1)
    top_p, top_class = output.topk(1, dim=1)
    equals = top_class == labels.long().view(*top_class.shape)

    actual_list.extend(labels.long().view(*top_class.shape))
    predictd_list.extend(top_class)




cm = confusion_matrix(actual_list, predictd_list)


ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');



     
    








