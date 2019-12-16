import torch
import torch.nn.functional as F
import configparser
import matplotlib.pyplot as plt
import os
import numpy as np
plt.figure(figsize=(20,5))

import ast

import sys
#to import parent-level modules
os.chdir('Temporal-Based Approach')
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
output_size = int(config['Common']['output_size'])
test_dir =  config['Test']['test_dir']
weights_loc = config['Common']['weights_loc']


initial_seq = torch.Tensor(ast.literal_eval(config['Save']['initial_seq']))
# max_note = int(config['Save']['max_note'])
# min_note = int(config['Save']['min_note'])

#check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

initial_seq = initial_seq.to(device)

#load the weights of the LSTM model
model = StackedLSTM(input_size,hidden_size,num_layer,output_size, batch_size)
model.load_state_dict(torch.load('{}'.format(weights_loc)))

#set the model in evaluation mode
model.eval()
model.to(device)



test_list = os.listdir(test_dir)
for each_test_file in test_list:

    test_file_path = os.path.join(test_dir,each_test_file).replace('\\','/')
    testing_data = np.array(normalize_testdata(test_file_path, 50, 89))

    predicted_notes_list = prediction_with_influence(model, testing_data, initial_seq)

    print(predicted_notes_list)

    #convert tensor to list
    for i in range(len(predicted_notes_list)):
        predicted_notes_list[i]=predicted_notes_list[i].detach().cpu().numpy().tolist()


    postprocessing = PostProcessing()
    postprocessing.stich_notes(predicted_notes_list)
    postprocessing.music_generation(testing_data*89, each_test_file)


    
