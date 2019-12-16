import torch
import torch.nn.functional as F
import configparser
import matplotlib.pyplot as plt
import os
import numpy as np
import ast
import sys

plt.figure(figsize=(20,5))

#to import parent-level modules
os.chdir('approach1')
sys.path.append('..')

from model.StackedLSTM import StackedLSTM
from utils.normalize_testdata import normalize_testdata
from influence import prediction_with_influence
from postprocessing.postprocessing import PostProcessing


#read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

batch_size = int(config['Test']['test_batch_size'])
input_size = int(config['Common']['input_size'])
hidden_size = int(config['Common']['hidden_size'])
num_layer = int(config['Common']['num_layer'])
sequence_length = int(config['Common']['sequence_length'])
weights_loc = config['Common']['weights_loc']
output_size = int(config['Common']['output_size'])
test_dir =  config['Test']['test_dir']

initial_seq = ast.literal_eval(config['Save']['initial_seq'])
int2note = ast.literal_eval(config['Save']['int2note'])
max_note = int(config['Save']['max_note'])
min_note = int(config['Save']['min_note'])

#check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#load the weights of the LSTM model
model = StackedLSTM(input_size,hidden_size,num_layer,output_size, batch_size)
model.load_state_dict(torch.load('{}'.format(weights_loc)))

#set the model in evaluation mode
model.eval()
model.to(device)


test_list = os.listdir(test_dir)
for each_test_file in test_list:
    '''
    
    '''

    test_file_path = os.path.join(test_dir,each_test_file).replace('\\','/')
    testing_data = normalize_testdata(test_file_path, min_note, max_note)

    predicted_notes = prediction_with_influence(model,testing_data,sequence_length,int2note,initial_seq, max_note, min_note)

    #save the graph
    plt.plot(np.array(testing_data)*max_note, label='data')
    plt.plot(predicted_notes, label = 'predicted')
    plt.legend(['Data', 'Predicted Notes'], loc='upper left')
    plt.ylabel('Pitch 0-127')
    plt.xlabel('Time Stamp')
    plt.savefig('output/{}.png'.format(each_test_file))
    plt.clf()

    #convert predicted notes to music file
    PostProcessing().generate_midi_file('output/{}.midi'.format(each_test_file), predicted_notes)