from .convert_to_midi import ConvertToMIDI

class PostProcessing():

    def __init__(self):
        self.pitch=[]
        self.duration=[]
        self.volume=[]
        self.start_time=[]
        self.start_time_index=[]

        self.convert_to_midi = ConvertToMIDI()

    def stich_notes(self, pred_list):
        '''
        Stich the predicted notes along with the trajactory data progressively

        Parameters:
            - pred_list (list) :

        Returns:
            - None

        '''
        counter=0
        for i in range(len(pred_list)):
            self.pitch.append(int(pred_list[i][0]*90))
            self.volume.append(int(pred_list[i][-1]*128))
            duration_list=pred_list[i][1:17]
            start_time_list=pred_list[i][17:97]
            
            for durations in range(len(duration_list)):
                if (duration_list[durations]==1):
                    self.duration.append((durations+1)*0.25)

            for start_time_value in range(len(start_time_list)): 
                if(start_time_list[start_time_value]==1):
                    if(counter>=1):
                        self.start_time_index.append(start_time_value)
                        if(start_time_value>self.start_time_index[counter-1]):
                            self.start_time.append(self.start_time[counter-1]+(start_time_value-self.start_time_index[counter-1])*0.25)
                        elif(start_time_value==self.start_time_index[counter-1]):
                            self.start_time.append(self.start_time[counter-1])
                        else:
                            self.start_time.append(self.start_time[counter-1]+((start_time_value)*0.25)+(80-self.start_time_index[counter-1])*0.25)
                    else:
                        self.start_time_index.append(start_time_value)
                        self.start_time.append((start_time_value)*0.25)
                    
            counter+=1




    def music_generation(self, testing_data, file_name):
        '''
        

        Parameters:
            - testing_data (list) :
            - file_name (string) :

        Returns:
            - None

        '''
        print("Entered Here")
        for i in range(len(self.duration)):
            self.duration[i] = self.duration[i]+0.5

        self.pitch = self.convert_to_midi.normalization(self.pitch,90,50)

        start_time = self.convert_to_midi.convert_start_time(self.start_time)
        result = self.convert_to_midi.music_generation(self.pitch,testing_data,self.duration,self.volume,self.start_time, file_name)

# if __name__ == "__main__":
    