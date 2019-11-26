import pandas as pd
import numpy as np
import sklearn.preprocessing
import csv
import os
import math


def normalize(filepath, min_note, max_note):
    
    new_population_list = []
    my_list=[]
    normalized_list=[]
    
    
    data = pd.read_csv(filepath)
    df=pd.DataFrame(data)
    #df = df.drop(df.index[0:5])
    df.dropna(inplace = True)

    
    
    population_value = df.iloc[:,1:2]
    
    my_list = df["Adj Close"].values
    #print(my_list)
    
    minimum_pop = my_list.min()
    maximum_pop = my_list.max()
    
    OldRange = (maximum_pop - minimum_pop)  
    NewRange = (max_note - min_note)
    
    for i in range(len(my_list)):
        new_population_list.append(math.floor((((my_list[i] - minimum_pop) * NewRange) / OldRange) + min_note))
    #print(new_population_list)
    


    for value in range(len(new_population_list)):
        normalized_list.append((new_population_list[value] - min_note)/(max_note - min_note))
    
    return(normalized_list)
    
if __name__ == '__main__':
    result=normalize("D:\\Prem\\Sem1\\MM in AI\\Project\\Project\\Sonification-using-Deep-Learning\\Dataset\\Test_Data\\trading.csv", 50,90)
    for i in range(len(result)):
        result[i]=(result[i]*(90-50)+50)/90
    print(result)
   
    
    
    
 




