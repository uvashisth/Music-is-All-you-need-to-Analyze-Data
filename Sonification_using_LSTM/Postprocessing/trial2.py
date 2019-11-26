from music21 import *
# prepare music data

def process_data(songs):
    whole_data = []
    for song in songs:
        midi_data = converter.parse(song).flat
        song_data = []
        prev_offset = -1
        for element in midi_data:
            if isinstance(element, note.Note):
                if element.offset != prev_offset:
                    song_data.append([element.pitch.nameWithOctave, 
                                      element.quarterLength])
                else:
                    if len(song_data[-1]) < 4:
                        song_data[-1].append(element.pitch.nameWithOctave)   
                        song_data[-1].append(element.quarterLength)       
                prev_offset = element.offset
            elif isinstance(element, chord.Chord):
                pitch_names = '.'.join(n.nameWithOctave for n in element.pitches)
                if element.offset != prev_offset:
                    song_data.append([pitch_names, element.quarterLength])
                else:
                    if len(song_data[-1]) < 4:
                        song_data[-1].append(pitch_names)   
                        song_data[-1].append(element.quarterLength)      
                prev_offset = element.offset
        for item in song_data:
            if len(item) < 4:
                item.append(None)
                item.append(None)
        whole_data.append(song_data)
    return whole_data
if __name__ == '__main__':
    result=process_data(["D:\\Downloads\\trial.mid"])
    print(result)