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


X, y = get_dataset(df_name)
st.write("Shape of Dataset", X.shape)
st.write("Number of Classes", len(np.unique(y)))

def add_parameter(classifier_name):
    params=dict()
    if classifier_name=="KNN":
        #number of neighbours
        K = st.sidebar.slider("Number of neighbors",1,15)        
        params["K"] = K
    elif classifier_name=="SVM":
        #Regularization parameter
        C =  st.sidebar.slider("Regularization parameter",0.01,10.0)   
        degree = st.sidebar.slider("Degree",1,10)
        kernel = st.sidebar.selectbox("Kernel",["linear", "poly", "rbf", "sigmoid"])
        params["C"] = C
        params["degree"] = degree
        params["kernel"] = kernel
    else:
        #max depth
        max_depth = st.sidebar.slider("Max Depth",2,15)
        n_estimators = st.sidebar.slider("Number of Estimators",1,100)
        criterion = st.sidebar.selectbox("Criterion",["gini", "entropy", "log_loss"])
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators           
        params["criterion"] = criterion           
    return params
    
add_parameter(classifier_name)
