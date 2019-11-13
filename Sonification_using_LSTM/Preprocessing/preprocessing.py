#Standard library modules
import glob
import sys
import os
import logging
from fnmatch import fnmatch

#Third Party imports
from music21 import *
import numpy as np
import torch 
from sklearn import preprocessing

#Local Modules
from util.midi_class_mapping import MidiClassMapping
from util.midi_notes_mapping import MidiNotesMapping


class PreprocessingTrainingData():
    """
    This class is created to preprocess the training data and return the input, output and min and max midi values which will be required for training
    """
    def __init__(self,sequence_length=50):
        self.sequence_length=sequence_length

    #Create a Logging File
    logging.basicConfig(filename="test.log", level=logging.DEBUG)
    """
    This function is to extract the notes from the midi file 
    Input Parameters: Absolute File path of the midi file
    Output Parameters: List of notes 
    """
    def extract_notes(self,file_path):
        #Intializing empty set
        notes = {}
        #Check if the input path is a file or not
        notes = self.get_notes(file_path)
        #Return the list of notes
        return notes
    
    """
    This function is to read midi file and get notes
    Input Parameters:Absolute File path of the midi file
    Output Parameters:List of notes 
    """
    def get_notes(self,filename):
        #Read the midi file
        midi = converter.parse(filename)
        notes_i = []
        notes_to_parse = None
        #Logging file
        logging.debug("File that is being parsed currently is {}".format(filename))
        
        try: 
            # Extracting the instrument parts
            notes_to_parse = midi[0].recurse()
        
        except: 
            # Extracting the notes in a flat structure
            notes_to_parse = midi.flat.notes
        #Iterate through each and every element in the notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # Taking the note
                notes_i.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # Taking the note with the highest octave.
                notes_i.append(str(element.pitches[-1])) 
        return notes_i
    
    """
    This function to calculate the count of unique number of notes
    Input Parameters: List of all notes from file
    Output Parameters: Number of unique number of notes
    """
    def number_of_output_notes_generated(self,notes):
        #Convert 2D list into 1D list
        all_notes=[]
        #Iterate through the 2D list
        for item in notes:
            all_notes.extend(item)
        #Number of unique notes
        number_of_output_notes=len(set(all_notes))
        return number_of_output_notes
    """
    This function is to normalize data 
    Input Parameters: List of input values
    Output Parameters: List of normalized data
    """
    def normalize_data(self,list_of_input_values,min_value,max_value):
        
        #Normalize each value of the list
        for i in range(len(list_of_input_values)):
            list_of_input_values[i]=(list_of_input_values[i]-min_value)/(max_value-min_value)
        return list_of_input_values
    """
    This function is to generate training data i.e model input,output,max value,min value
    Input Parameters: Set of input notes read from midi files
    Output Parameters: Network Input,Network Output, Max midi Value,Min midi value
    """
    def generate_training_data(self,notes):
        
        #Generate a flat list of input notes
        notes_from_training_data = []
        
        for item in notes:
            notes_from_training_data.extend(item)

        # get all right hand note names
        right_hand_notes = sorted(set(item for item in notes_from_training_data))
        #Get note to midi number mapping
        note_to_midi_number_mapping=MidiNotesMapping().get_midi_number_notes_mapping("../A.txt")
        #Get maximum and minimum midi number values
        _,int_to_note,max_midi_value,min_midi_value=MidiClassMapping().midi_notes_to_class_mapping(right_hand_notes,note_to_midi_number_mapping)
        
        
        network_input = []
        network_output = []
        for song in notes:
            for i in range(0, len(song) - self.sequence_length, 1):                
                sequence_in = song[i:i + self.sequence_length]           
                sequence_out = song[i + self.sequence_length]
                for notes in range(len(sequence_in)):
                    for key,value in note_to_midi_number_mapping.items():
                        if  str(sequence_in[notes]) in value:
                            sequence_in[notes]=key
                
                for key,value in note_to_midi_number_mapping.items():
                    if  str(sequence_out) in value:
                        sequence_out=key    
                network_input.append(sequence_in)
                network_output.append(int(sequence_out) )
        #Check if length of input and output is same
        assert len(network_input) == len(network_output), len(network_input)
        #Number of input batches
        n_patterns = len(network_input)
        #Normalize the input data
        for i in range(len(network_input)):    
            network_input[i]=self.normalize_data(network_input[i],min_midi_value,max_midi_value)
        #Normalize the output data    
        network_output=self.normalize_data(network_output,min_midi_value,max_midi_value)
        #Converting 2d list to 2d numpy array
        network_input=np.array(network_input)
        #Reshaping the 2d numpy array to 3d array
        network_input = np.reshape(network_input, (n_patterns, self.sequence_length, 1))        
        return (network_input, network_output,max_midi_value,min_midi_value,int_to_note)
    """
    This is the main function which has to be called it acts like a wrapper function
    Input Parameters:
    Output Parameters:
    """
    def preprocess_notes(self,path):
        pattern = "*.mid"
        notes=[]
        if not path.endswith(".mid"):
            for path, subdirs, files in os.walk(path):
                for name in files:
                    if fnmatch(name, pattern):
                        notes.append(self.extract_notes(os.path.join(path, name)))
        else:        
            notes.append(self.extract_notes(path))
        number_of_output_notes=self.number_of_output_notes_generated(notes)
        network_input,network_output,max_midi_number,min_midi_number,int_to_note=self.generate_training_data(notes)
        for i in range(len(network_input)):
            for j in range(len(network_input[i])):
                temp=[]
                
                temp.append((network_input[i][j]))
                network_input[i][j]=temp
        network_input = np.asarray(network_input,dtype=np.float32)
        network_input=torch.tensor(network_input)
        network_output=torch.tensor(network_output)

        return network_input,network_output,max_midi_number,min_midi_number,int_to_note


if __name__=="__main__":
    network_input,network_output,max_midi_number,min_midi_number,int_to_note=PreprocessingTrainingData().preprocess_notes("D:\\Prem\\Sem1\\MM in AI\\Project\\Project\\Sonification-using-Deep-Learning\\Dataset\\Clementi dataset\\Clementi dataset\\clementi_opus36_1_1.mid")
    