import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
        random_state = st.sidebar.number_input("Random State",0,1000000)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators           
        params["criterion"] = criterion     
        params["random"] = random_state      
    return params
    
params = add_parameter(classifier_name)

def get_classifier(classifier_name,params):
    if classifier_name=="KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name=="SVM":
        classifier = SVC(C=params["C"],degree=params["degree"],kernel=params["kernel"])
    else:
        classifier=RandomForestClassifier(max_depth=params["max_depth"],n_estimators=params["n_estimators"],criterion=params["criterion"],random_state=params["random"])
    return classifier

classifier = get_classifier(classifier_name,params)

#CLASSIFICATION
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=12345)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test,y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")
