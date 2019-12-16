import torch
import torch.nn.functional as F

#Predicting
def prediction_with_influence(model, influence,initial_seq):

    predicted_notes = []
    
    print(influence.shape)
    print(initial_seq.shape)

    test_seq = initial_seq.clone()
    
    for i in range(len(influence)):
                
        
        test_seq[0][-1][0]=float(influence[i])
        
                
        test_output = model.forward(test_seq)

        logis = F.sigmoid(test_output[0])
        class_pred=test_output[0][0].reshape(-1)
                
        length_duration = torch.zeros(16) #Number of different types of durations
        length_duration[logis[1:17].max(0)[1].item()] = 1 #Whichever length had highest probability is the one chosen
        
        length_start = torch.zeros(80) #Number of different types of durations
        length_start[logis[17:97].max(0)[1].item()] = 1 #Whichever length had highest probability is the one chosen
        
        vel = test_output[0][-1].reshape(-1)
        
        con = torch.cat((class_pred.cuda(), length_duration.cuda(), length_start.cuda(),vel.cuda()))

        predicted_notes.append(con)

        
        test_seq[0][-1] = con        

        for p,each_col in enumerate(test_seq[0]):
            if p + 2 > len(test_seq[0]):
                break
            test_seq[0][p] = test_seq[0][p+1]
                   
    return predicted_notes