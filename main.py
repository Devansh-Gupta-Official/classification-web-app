import streamlit as st
from sklearn import datasets
import numpy as np

st.title("Explore Different Classifiers")
st.write("### Which one is the best?")

df_name = st.sidebar.selectbox("Select Dataset",["Iris","Breast Cancer","Wine Dataset"])

classifier_name = st.sidebar.selectbox("Select Classifier",["KNN","SVM","Random Forest"])

def get_dataset(df_name):
    if df_name=="Iris":
        data = datasets.load_iris()
    elif df_name =="Breast Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X = data.data
    y = data.target
    return X, y



