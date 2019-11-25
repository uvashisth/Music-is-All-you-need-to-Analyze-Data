from mido import Message, MidiFile, MidiTrack
mid = MidiFile('output.mid')
midi1 = MidiFile()
track1 = MidiTrack()


midi1.tracks.append(track1)
track1.append(Message('program_change', program=12, time=0,channel=1))

counter=1
for i, track in enumerate(mid.tracks):
    print("Entered")
    print('Track {}: {}'.format(i, track.name))
    
    for msg in track:
        # print(msg)
        msg=str(msg).replace("<","")
        msg=str(msg).replace(">","")
        msg=str(msg).split(" ")
        if(msg[0]=='note_on' and counter==1):
            note=0
            for i in range(len(msg)):
                if "note=" in msg[i]:
                    note=msg[i].replace("note=","")
                    print(note)
            track1.append(Message("note_on",note=int(note), velocity=64,time=320,volume=int(note)))
            counter+=1
        elif(msg[0]=="note_on"):
            note=0
            for i in range(len(msg)):
                if "note=" in msg[i]:
                    note=msg[i].replace("note=","")
            track1.append(Message("note_on",note=int(note), velocity=(64),time=320,volume=int(note)))
            counter+=1
        elif(msg[0]=="note_off"):
            note=0
            for i in range(len(msg)):
                if "note=" in msg[i]:
                    note=msg[i].replace("note=","")
            track1.append(Message("note_off",note=int(note), velocity=127,time=320,volume=int(note)))
            counter+=1
        else:
            print("Error")
track1.append(Message('program_change', program=12, time=0,channel=2))

midi1.save('new_song.mid')


        
        