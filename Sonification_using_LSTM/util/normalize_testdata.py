import pandas as pd
import math

def normalize_testdata(filepath, max_note, min_note):
    '''


    Paramters:

        filepath (string) : 
        max_note (int): 
        min_note (int):


    Returns: 
        A list containing normalized values
    ''' 

    normalized_test=[]
    
    test_df = pd.read_csv(filepath)
    test_df.dropna(inplace = True)

    test_values = test_df.iloc[:,0].values
    
    max_test = test_values.max()
    min_test = test_values.min()

    old_range = (max_test - min_test)  
    new_range = (max_note - min_note)
    

    #scale the data to a new range and then normalize it
    for i in range(len(test_values)):
        normalized_test.append(math.floor((((test_values[i] - min_test) * new_range) / old_range) + min_note)/max_note)


    return normalized_test
    
   
    
    # if __name__ == '__main__':
    #     pass
    
 




