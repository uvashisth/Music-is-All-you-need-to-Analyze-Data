import pretty_midi
import matplotlib.pyplot as plt
# Load MIDI file into PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI('D:\\Prem\\Sem1\\MM in AI\\Project\\Project\\Sonification-using-Deep-Learning\\Dataset\\Clementi dataset\\Clementi dataset\\clementi_opus36_1_1.mid')
# Print an empirical estimate of its global tempo
print(midi_data.get_beats())
notes=[]
start_time=[]
end_time=[]
pitch=[]
velocity=[]
for instrument in midi_data.instruments:
    # Don't want to shift drum notes
    for note in instrument.notes:
       notes.append(note.pitch)
       start=str(note).split("(")[1].split(")")[0].split(",")[0].split("=")[1]
       start_time.append(start)
       end=str(note).split("(")[1].split(")")[0].split(",")[1].split("=")[1]
       end_time.append(end)
       pitch1=str(note).split("(")[1].split(")")[0].split(",")[2].split("=")[1]
       pitch.append(pitch1)
       velocity1=str(note).split("(")[1].split(")")[0].split(",")[3].split("=")[1]
       velocity.append(velocity1)

print(pitch)
print(velocity)
plt.plot(start_time,pitch)
plt.plot(end_time,pitch)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Pitch")
plt.show()