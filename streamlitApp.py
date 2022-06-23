import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json
import joblib
from sklearn import preprocessing 


def showModelEval(modelDataPathR):
    
    #show scoring
    modelEvaluationDataPath = modelDataPathR + ".json"
    with open(modelEvaluationDataPath) as file:
        modelEvaluationData = json.load(file)
    
    
    st.write("Achieved accuracy with final test on the test dataset")
    acc = str(round(modelEvaluationData["acc"], 4))
    acc = '<p style="font-size: 32px; color: Green">{}</p>'.format(acc)
    st.markdown(acc, unsafe_allow_html=True)
    
    st.write("Achieved Recall with final test on the test dataset")
    rec = str(round(modelEvaluationData["rec"], 4))
    rec = '<p style="font-size: 32px; color: Green">{}</p>'.format(rec)
    st.markdown(rec, unsafe_allow_html=True)
    
    #show confusion matrix
    modelCm = [[modelEvaluationData["tn"], modelEvaluationData["fp"]],
               [modelEvaluationData["fn"], modelEvaluationData["tp"]]]

    modelCm = np.array(modelCm)
    
    fig, ax = plt.subplots(figsize = (3, 3))
    ax.matshow(modelCm)
    
    for (i, j), value in np.ndenumerate(modelCm):
        ax.text(j, i, "{}".format(value), ha='center', va='center')

    buf = BytesIO()
    fig.savefig(buf, format = "png")
    st.image(buf)

def getUserInput():
    userData = []
    labels = ["Age", "Sex",
              "Chest pain type: 1 for typical angina,   2 for atypical angina,   3 for non-anginal pain,   4 for asymptomatic ", "Resting blood pressure (in mm Hg on admission to the hospital)",
              "Serum cholestoral in mg/dl", "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
              "Resting electrocardiographic results,   0 for normal,   1 for having ST-T wave abnormality and   2 for showing probable or definite left ventricular hypertrophy by Estes' criteria", 
              "Maximum heart rate achieved",
              "Exercise induced angina (1 = yes; 0 = no)","Oldpeak = ST depression induced by exercise relative to rest",
              "The slope of the peak exercise ST segment,   1 for upsloping,   2 for flat,   3 for downsloping ",
              "Number of major vessels (0-3) colored by flourosopy","Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
    for label in labels:
        userData.append(float(st.text_input(label, 0)))
    return processUserInput(np.array(userData).reshape(1, 13))


def getModelPrediction(modelPath, userData):
    model = joblib.load(modelPath)
    return model.predict(userData)


def processUserInput(userData):
    cleanedDataset = pd.read_csv(projectDirPath + "/data/cleaned/X_cleaned.csv", sep=',', header=None)
    cleanedDataset = np.vstack([cleanedDataset,userData])
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max = min_max_scaler.fit_transform(cleanedDataset)
    return min_max[-1,:].reshape(1, 13)
    
    
#conventional way of getting the file path doesn't work with streamlit so this is used
projectDirPath = os.path.dirname(os.path.abspath(__file__))

#General info
st.header("Heart disease diagnosis")
st.write("In this web app you can enter your data and determine if you potentialy have heart disease." + 
         "Five machine learning models were trained and you'll get diagnosis from each one. " +
         "Under the input form you can look at the model evaluation data for test data set.")
st.markdown("__- Logistic Regression__")
st.markdown("__- Support Vector Machine__")
st.markdown("__- K Nearest Neighbors__")
st.markdown("__- Random forest classifier__")
st.markdown("__- Gaussian Nayve Bayes__")


#take user input and get prediction for every model that was chosen
st.header("Enter your data:")


with st.form("user input"):
    userInput = getUserInput()
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.subheader("Logistic regression prediction:")
        logPath = projectDirPath + "/models/logisticReg.joblib"
        logPred = getModelPrediction(logPath, userInput)
        #st.write(userInput)
        #st.write(logPred)
        if (logPred == 1):
            st.markdown('<p style="font-size: 28px; color: Red">Possibility of heart disease</p>', unsafe_allow_html=True)
            st.write("Possibility of heart disease")
        else:
            st.markdown('<p style="font-size: 28px; color: Green">No heart disease</p>', unsafe_allow_html=True)
            st.write("No heart disease")
            
        st.subheader("Support vector classifier prediction:")
        svcPath = projectDirPath + "/models/svc.joblib"
        svcPred = getModelPrediction(svcPath, userInput)
        if (logPred == 1):
            st.markdown('<p style="font-size: 28px; color: Red">Possibility of heart disease</p>', unsafe_allow_html=True)
            st.write("Possibility of heart disease")
        else:
            st.markdown('<p style="font-size: 28px; color: Green">No heart disease</p>', unsafe_allow_html=True)
            st.write("No heart disease")
        
        st.subheader("K nearest neighbors prediction:")
        knnPath = projectDirPath + "/models/knn.joblib"
        knnPred = getModelPrediction(knnPath, userInput)
        if (logPred == 1):
            st.markdown('<p style="font-size: 28px; color: Red">Possibility of heart disease</p>', unsafe_allow_html=True)
            st.write("Possibility of heart disease")
        else:
            st.markdown('<p style="font-size: 28px; color: Green">No heart disease</p>', unsafe_allow_html=True)
            st.write("No heart disease")
        
            
        st.subheader("Random forest classifier prediction:")
        ranForPath = projectDirPath + "/models/randomForest.joblib"
        ranForPred = getModelPrediction(ranForPath, userInput)
        if (ranForPred == 1):
            st.markdown('<p style="font-size: 28px; color: Red">Possibility of heart disease</p>', unsafe_allow_html=True)
            st.write("Possibility of heart disease")
        else:
            st.markdown('<p style="font-size: 28px; color: Green">No heart disease</p>', unsafe_allow_html=True)
            st.write("No heart disease")
        
        st.subheader("Gaussian Nayve Bayes prediction:")
        gaussNBPath = projectDirPath + "/models/gaussianNB.joblib"
        gaussNBPred = getModelPrediction(gaussNBPath, userInput)
        if (gaussNBPred == 1):
            st.markdown('<p style="font-size: 28px; color: Red">Possibility of heart disease</p>', unsafe_allow_html=True)
            st.write("Possibility of heart disease")
        else:
            st.markdown('<p style="font-size: 28px; color: Green">No heart disease</p>', unsafe_allow_html=True)
            st.write("No heart disease")
        
        

st.header("Logistic Regression Analysis")
st.write("Two models were trained, the one with better recall was chosen"
        + " because for this specific task recall is more important than accuracy")


st.header("Support Vector Classifier")
st.write("One model with linear kernel was trained")
showModelEval(projectDirPath + "/models/logisticRegression")



st.header("Support Vector Classifier")
st.write("One model with linear kernel was trained")
showModelEval(projectDirPath + "/models/svc")


st.header("K Nearest Neighbors Analysis")
st.write("Three models were trained, with 3, 4 and 5 neighbors. " + 
         "The one with the highest recall was chosen")

col1Knn, col2Knn, col3Knn = st.columns(3)

with col1Knn:
    st.subheader("K-NN, 3 neighbors")
    showModelEval(projectDirPath + "/models/knn1")
with col2Knn:
    st.subheader("K-NN, 4 neighbors")
    showModelEval(projectDirPath + "/models/knn2")
with col3Knn:
    st.subheader("K-NN, 5 neighbors")
    showModelEval(projectDirPath + "/models/knn3")
st.markdown('<p style="font-size: 28px;">4 neighbors ' +
            'were chosen for its best accuracy and recall</p>', unsafe_allow_html=True)

st.header("Random forest classifier")
showModelEval(projectDirPath + "/models/randomForest")


st.header("Gaussian Nayve Bayes")
showModelEval(projectDirPath + "/models/gaussianNB")




#Dataset overview
st.header("Original dataset overview")

datasetPath = projectDirPath + "/data/raw/mixed_data.data"
dataset = pd.read_csv(datasetPath)

newNames = {"63.0" : "Age",
            "1.0" : "Sex",
            "1.0.1" : "Chest pain type",
            "145.0" : "Resting blood pressure [mm Hg]",
            "233.0" : "Serum cholestoral [mg/dl]",
            "1.0.2" : "Fasting blood sugar ",
            "2.0" : "Resting electrocardiographic results",
            "150.0" : "Maximum heart rate achieved",
            "0.0" : "Exercise induced angina",
            "2.3" : "oldpeak ",
            "3.0" : "slope",
            "0.0.1" : "number of major vessels",
            "6.0" : "thal",
            "0" : "diagnosis of heart disease "}
dataset = dataset.rename(columns = newNames, inplace = False)
st.write(dataset)
st.write("Class: >= 1 - Possible presence of heart disease, 0 - No heart disease")     


