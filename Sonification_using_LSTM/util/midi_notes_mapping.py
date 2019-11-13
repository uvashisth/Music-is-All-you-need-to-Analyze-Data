import os
class MidiNotesMapping():
    def get_midi_number_notes_mapping(self,file_name):
        note_to_midi_mapping = {}
        with open(file_name) as f:
            for line in f:
                (midi_value, notes) = line.split()
                note_to_midi_mapping[int(midi_value)] = notes
        return note_to_midi_mapping
if __name__=="__main__":
    A=MidiNotesMapping()
    value=A.get_midi_number_notes_mapping("D:/Prem/Sem1/MM in AI/Project/Project/Sonification-using-Deep-Learning/Sonification using LSTM/A.txt")
    print(value)