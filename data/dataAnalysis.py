import pandas as pd
import os

sick_count = 0
healthy_count = 0

projectDirPath = os.path.abspath('../')

raw_data = pd.read_csv(projectDirPath + "/data/raw/mixed_data.data", sep = ',', header=None)

y = raw_data.iloc[:, -1].values

for i in range(0, len(y)):
    if y[i] == 1:
        sick_count+= 1
    else:
        healthy_count+= 1

raw_data = raw_data.to_numpy()

print("In this dataset, there are ", len(raw_data), " diagnosis.")
print("There are" , healthy_count , ", or " , format(healthy_count/(sick_count+healthy_count)*100, ".2f") , "% , diagnosis with class 0 or diagnosis without heart disease")
print("There are" , sick_count , ", or " , format(sick_count/(sick_count+healthy_count)*100, ".2f") , "% , diagnosis with class 1 or heart disease diagnosis")

complete_rows = 0
row_is_complete = True
for i in range(0, len(raw_data)):
    for j in range(0, 14):
        if raw_data[i][j] == '?':
            row_is_complete=False     
    if row_is_complete:
        complete_rows += 1
    else:
        row_is_complete = True
    
print("There are ", complete_rows, " or ",format(complete_rows/len(raw_data)*100, ".2f"), "% complete diagnosis (rows without missing data)")
    

missing_values_count = 0
for i in range(0, 14):
    for j in range(0, len(raw_data)):
        if raw_data[j][i] == '?':
            missing_values_count += 1
    print("In column number: ", i+1, ", There are: ", missing_values_count, " or ", format(missing_values_count/len(raw_data)*100, ".2f"),"% missing values")
    missing_values_count = 0
        
    
    