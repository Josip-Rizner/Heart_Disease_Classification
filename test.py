import streamlit as st
import pandas as pd
import os
import joblib
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def showModelEval():
    st.write("Model validation")
    """
    fig, ax = plt.subplots(figsize = (3, 3))
    matrix = np.zeros((2, 2))
    ax.matshow(matrix)
    buf = BytesIO()
    fig.savefig(buf, format = "png")

    col1, col2 = st.columns(2)
    col1.image(buf)
    col2.image(buf)
    """

#conventional way of getting the file path doesn't work with streamlit so this is used
projectDirPath = os.path.dirname(os.path.abspath(__file__))

#General info
st.header("Breast cancer tumor classification")
st.write("In this app you can view the analysis of breast cancer tumor" + 
         " classification problem using three machine learning models:")
st.markdown("__- Logistic Regression__")
st.markdown("__- Support Vector Machine__")
st.markdown("__- K Nearest Neighbors__")
st.write("it is also possible to enter your own values of features, and" +
         " get the prediction using all three models")

#Dataset overview
st.header("Original dataset overview")

datasetPath = projectDirPath + "\data\\raw\\cleveland_only.data"
dataset = pd.read_csv(datasetPath)

newNames = {"1000025" : "Sample code number",
            "5" : "Clump Thickness",
            "1" : "Uniformity of Cell Size",
            "1.1" : "Uniformity of Cell Shape",
            "1.2" : "Marginal Adhesion",
            "2" : "Single Epithelial Cell Size",
            "1.3" : "Bare Nuclei",
            "3" : "Bland Chromatin",
            "1.4" : "Normal Nucleoli",
            "1.5" : "Mitoses",
            "2.1" : "class"}
datasetNewNames = dataset.rename(columns = newNames, inplace = False)
st.write(datasetNewNames)
st.write("sample code number - irrelevant for classification (removed in preprocessed data)")
st.write("Class: 4 - malignant, 2 - benign")

#Logistic Regression analysis
st.header("Logistic Regression Analysis")
st.write("Two models were trained, the one with better recall was chosen"
         + " because for this specific task recall is more important that accuracy")
st.subheader("Logistic Regression - Linear border")

