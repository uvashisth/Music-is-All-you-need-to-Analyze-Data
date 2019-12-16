from midiutil import MIDIFile

class PostProcessing():
    '''

    '''

    def __init__(self,track=0,channel=0,time=0,duration=1,tempo=60,volume=100):
        '''
        Constructor of the class

        Parameters:
            - track (int) :
            - channel (int) :
            - time (int) :
            - duration (int) :
            - tempo (int) :
            - volume (int) :

        Returns:
            - An object of type Postprocessing 

        '''
        self.track=track
        self.channel=channel
        self.time=time
        self.duration=duration
        self.tempo=tempo
        self.volume=volume

    def generate_midi_file(self,file_path,midi_number_list):
        '''
        Divide the dataset into train/val

        Parameters:
            - file_path (float) :
            - midi_number_list (list) :

        Returns:
            - two generators

        '''

        output_MIDI_file = MIDIFile(1)
        output_MIDI_file.addTempo(self.track, self.time, self.tempo)
        
        for i, pitch in enumerate(midi_number_list):
            self.volume=int(((pitch-50)/(90-50))*100)
            output_MIDI_file.addNote(self.track, self.channel, pitch, self.time + i, self.duration, self.volume)
        with open(file_path,"wb") as output_file:
            output_MIDI_file.writeFile(output_file)