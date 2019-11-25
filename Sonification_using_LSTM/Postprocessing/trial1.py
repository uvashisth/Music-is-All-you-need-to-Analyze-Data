from midiutil import MIDIFile
import matplotlib.pyplot as plt
degrees  = [66, 75, 66, 75, 66, 75, 64, 76, 64, 76, 64, 76, 73, 81, 73, 81, 73, 81, 74, 81, 78, 79, 76, 74, 81, 78, 79, 76, 74, 86, 78, 86, 64, 69, 67, 69, 67, 71, 69, 71, 69, 81, 64, 69, 67, 69, 67, 71, 69, 71] # MIDI note number
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for pitch in degrees:
    volume=int(((pitch-50)/(90-50))*100)
    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    time = time + 1

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

plt.plot(degrees)
plt.show()