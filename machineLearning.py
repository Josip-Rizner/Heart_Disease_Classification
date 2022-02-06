import pandas as pd
import numpy as np




def cleanRowsWithMissingValues(raw_data):
    indexOfRowsWithMissingValues = [] 
    for row_index in range(1, raw_data.shape[0]):
        for column_index in range(raw_data.shape[1]):
            if(raw_data[row_index, column_index] == '?'):
                indexOfRowsWithMissingValues.append(row_index)
    raw_data = np.delete(raw_data, indexOfRowsWithMissingValues, axis=0)
    return raw_data

def splitData(cleaned_data):
    rows = cleaned_data.shape[0]
    rowsForTraining = round(rows*0.7)
    x_train = cleaned_data[0:rowsForTraining, 0:13]
    y_train = cleaned_data[0:rowsForTraining, 13]
    


raw_data = pd.read_csv("MachineLearning/data/cleveland.data", sep = ',', header=None)

raw_data = raw_data.values
cleaned_data = cleanRowsWithMissingValues(raw_data)
print(cleaned_data[0:5, 13])
splitData(cleaned_data)

