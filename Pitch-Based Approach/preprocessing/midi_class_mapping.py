
class MidiClassMapping():
    '''
    '''

    def __init__(self):

        self.midi_list=[]
        self.note_to_int={}
        self.int_to_note={}
        self.max_midi_value=0
        self.min_midi_value=0

    def midi_notes_to_class_mapping(self,notes,midi_notes_mapping):
        '''
        

        Parameters:
            - notes (float) :
            - midi_notes_mapping (torch.Tensor) :

        Returns:
            - two generators

        '''

        for note in notes:
            for midi,note_value in midi_notes_mapping.items():
                if str(note) in note_value:
                    self.midi_list.append(midi)
                    
        self.midi_list=sorted(self.midi_list)
        self.max_midi_value=self.midi_list[len(self.midi_list)-1]
        self.min_midi_value=self.midi_list[0]
        self.note_to_int = dict((note, number) for number, note in enumerate(self.midi_list))
        self.int_to_note={note:ii for ii,note in self.note_to_int.items()}

        return self.note_to_int,self.int_to_note,self.max_midi_value,self.min_midi_value