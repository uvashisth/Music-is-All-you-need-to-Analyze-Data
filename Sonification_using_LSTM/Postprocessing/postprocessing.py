from midiutil import MIDIFile
class PostProcessing():
    def __init__(self,track=0,channel=0,time=0,duration=1,tempo=60,volume=100):
        self.track=track
        self.channel=channel
        self.time=time
        self.duration=duration
        self.tempo=tempo
        self.volume=volume
    def generate_midi_file(self,file_path,midi_number_list):
        output_MIDI_file = MIDIFile(1)
        output_MIDI_file.addTempo(self.track, self.time, self.tempo)
        for i, pitch in enumerate(midi_number_list):
            self.volume=int(((pitch-50)/(90-50))*100)
            output_MIDI_file.addNote(self.track, self.channel, pitch, self.time + i, self.duration, self.volume)
        with open(file_path,"wb") as output_file:
            output_MIDI_file.writeFile(output_file)
if __name__=="__main__":
    A=PostProcessing(0,0,0,2,60,100)
    a=[66, 75, 66, 75, 66, 75, 64, 76, 64, 76, 64, 76, 73, 81, 73, 81, 73, 81, 74, 81, 78, 79, 76, 74, 81, 78, 79, 76, 74, 86, 78, 86, 64, 69, 67, 69, 67, 71, 69, 71, 69, 81, 64, 69, 67, 69, 67, 71, 69, 71]
    A.generate_midi_file("output.mid",a)
