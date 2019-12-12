import torch
import torch.nn as nn
import numpy as np
from utils.normalize_testdata import normalize_testdata
import seaborn as sns
import matplotlib.pyplot as  plt
import os
import logging
torch.set_printoptions(profile="full")
import pandas as pd
from model.StackedLSTM import StackedLSTM
from utils import split_test_val


#static parameters
batch_size = 30
train_size = 0.8
sequence_length=50
test_batch_size = 1
input_size = 1
hidden_size = 256
num_layer = 2
output_size = 38
clip = 3


#get data from preprocessing.py
dataset_path = os.path.join(os.path.abspath('..'),'Dataset\\Clementi dataset\\Clementi dataset' )
network_input,network_output,max_midi_number,min_midi_number,int_to_note = PreprocessingTrainingData().preprocess_notes(dataset_path)
network_input, network_output = network_input.cuda(), network_output.cuda()



#divide the dataset into train/val
train_loader, val_loader = split_test_val()





