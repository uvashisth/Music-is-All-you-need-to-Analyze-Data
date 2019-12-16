import torch
import torch.nn.functional as F

#Predicting
def prediction_with_influence(model,influence,sequence_length,int2note,initial_seq, max_note, min_note,test_batch_size = 1):
    
    predicted_notes = []
    initial_seq[0].extend([[0]]*len(influence))
    test_seq = torch.Tensor(initial_seq).cuda()
    
    for i in range(len(influence)):
        
        test_seq[0][sequence_length - 1 + i][0] = float(influence[i])
        
        test_slice = test_seq[0][i : i + sequence_length]        
        test_slice = test_slice.view(1, test_slice.shape[0], test_slice.shape[1])
        
        test_output = model.forward(test_slice)

        test_output = F.softmax(test_output, dim = 1)
        top_p, top_class = test_output.topk(1,dim =1)
        
        test_seq[0][sequence_length - 1 + i][0] = int2note[top_class.item()]/max_note
        
        predicted_notes.append(int2note[top_class.item()])
        
    return predicted_notes
