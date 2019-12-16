import torch
import torch.nn as nn
import numpy as np


class StackedLSTM(nn.Module):

    '''
    Create a StackedLSTM model
        - This class inherits from nnModule of Pytorch

    '''

    def __init__(self,input_size, hidden_size, num_layers, output_size, batch_size):

        '''
        Initializes StackedLSTM model

        Parameters :
            input_size (int): Input size of the LSTM Model,
            hidden_size (int): Number of nodes in the hidden layer,
            num_layers (int): Number of LSTM layers stacked together,
            output_size (int): Size of the output from the model,
            batch_size (int): Total number of elements in a batch

        Returns :
            An object of type StackedLSTM

        '''
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size

        self.layer1 = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, batch_first = True)
        self.layer2 = nn.LSTM(input_size = self.hidden_size, hidden_size = self.output_size,batch_first = True)
        
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        '''
        Forward Propagation

        Parameters :
            x (tensor): Input to the model of shape -> batch_size*sequence_length*input_size

        Returns:
            A tensor of shape -> batch_size*output_size

        '''
        
        output, _ = self.layer1(x)    

        #output = self.dropout(output)
        
        output, _ = self.layer2(output)
        
        # stack up lstm outputs
        output = output.contiguous().view(-1, self.output_size)
        
        output = self.dropout(output)
        output = self.linear(output)
        
        # reshape to be batch_size first
        output = output.view(self.batch_size, -1)

        # get last batch of labels
        output = output[:, -self.output_size:] 
        
        return output

    
def init_weights(m):
    '''
    Using Xavier initialization to initialize the weights of the Linear Layer. 
    
    Parameters:
        m (nn.Module): Model, whose weights needs to be initialized
    
    Returns:
        None
    '''
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight) 