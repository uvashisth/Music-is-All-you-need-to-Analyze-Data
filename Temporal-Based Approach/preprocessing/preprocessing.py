import pretty_midi
from itertools import repeat
import math
import os
from fnmatch import fnmatch


class PreprocessingTrainingDataM2():


    def __init__(self):
        self.notes=[]
        self.start_time=[]
        self.end_time=[]
        self.pitch_values=[]
        self.velocity_values=[]
        self.duration=[]


    def extract_notes(self, midi_data):
        '''
        

        Parameters:
            - midi_data (list) :

        Returns:
            - Tensors

        '''
        for instrument in midi_data.instruments:
                for note in instrument.notes:
                    self.notes.append(note.pitch)
                    self.start_time.append(str(note).split("(")[1].split(")")[0].split(",")[0].split("=")[1])
                    self.end_time.append(str(note).split("(")[1].split(")")[0].split(",")[1].split("=")[1])
                    self.duration.append(float(str(note).split("(")[1].split(")")[0].split(",")[1].split("=")[1])-float(str(note).split("(")[1].split(")")[0].split(",")[0].split("=")[1]))
                    self.velocity_values.append(str(note).split("(")[1].split(")")[0].split(",")[3].split("=")[1])
        

    def preprocess_notes(self,path):

        pattern = "*.mid"

        print(os.getcwd())
        if not path.endswith(".mid"):
            for path, subdirs, files in os.walk(path):
                for name in files:
                    if fnmatch(name, pattern):

                        file_path = os.path.join(path, name).replace('\\','/')
                        midi_data = pretty_midi.PrettyMIDI(file_path)
                        self.extract_notes(midi_data)
                        
        else:
            file_path = os.path.join(path, name).replace('\\','/')
            midi_data = pretty_midi.PrettyMIDI(file_path)
            self.extract_notes(midi_data)

       
        network_input=[]
        network_output=[]

        # # for i in range(len(notes)):
        for j in range(0, len(self.notes) - 50, 1):
            midivalue_in = self.notes[j:j +50]
            midivalue_out = self.notes[j + 50]
            duration_in = self.duration[j:j+50]
            duration_out=self.duration[j+50]
            velocity_in = self.velocity_values[j:j+50]
            velocity_out = self.velocity_values[j+50]
            start_in = self.start_time[j:j+50]
            start_out = self.start_time[j+50]
            temp_input=[]
            temp_output=[]
            for in1 in range(len(midivalue_in)):
                temp_in=[]
#                 temp_input_list=list(repeat(0, 40))
#                 temp_input_list[midivalue_in[in1]-50]=1
                temp_input_list=[]
                temp_input_list.append(midivalue_in[in1]/90)
                temp_in=temp_in+temp_input_list
                temp_duration=list(repeat(0, 16))
                duration1=math.ceil(duration_in[in1]/0.25)
                if(duration1>16):
                    temp_duration[15]=1
                else:
                    temp_duration[duration1-1]=1
                temp_start=list(repeat(0,80))
                start1=math.ceil(float(start_in[in1])%20/0.25)
#                 print(start1)
                if(start1>80):
                    start1[79]=1
                else:
                    temp_start[start1-1]=1
                temp_in=temp_in+(temp_duration)
#                 print(temp_start)
                temp_in=temp_in+temp_start
                temp_in.append(int(velocity_in[in1])/128)
                temp_input.append(temp_in)
            temp_duration=list(repeat(0, 16))
            duration1=math.ceil(duration_out/0.25)
            if(duration1>16):
                temp_duration[15]=1
            else:
                temp_duration[duration1-1]=1
            temp_start=list(repeat(0,80))
            start1=math.ceil(float(start_out)%20/0.25)
            if(start1>80):
                start1[79]=1
            else:
                temp_start[start1-1]=1
            
            temp_midi=[]
            temp_midi.append(midivalue_out/90)
            temp_output=temp_output+temp_midi
            
            temp_output=temp_output+temp_duration
            temp_output=temp_output+temp_start
            temp_output.append(int(velocity_out)/128)
            network_output.append(temp_output)
            network_input.append(temp_input)
        return network_input,network_output
            
#if __name__ == "__main__":
#     network_input,network_output=Preprocessing_New_Method().preprocessing()
#     print(len(network_output[0]))
