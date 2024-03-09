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
import seaborn as sns
from streamlit_lottie import st_lottie
import simplejson
from streamlit_feedback import streamlit_feedback  #from trubrics
from streamlit_star_rating import st_star_rating

st.set_page_config(
    page_title="Classification",
    page_icon=":question:",
    initial_sidebar_state="auto",
    layout="wide"
)


hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# st.markdown(""" <style> .font {
# font-size:50px ; font-family: 'Lucida Console'; color: #F2F2F2;} 
# </style> """, unsafe_allow_html=True)
# st.markdown('<h1 class="font">Explore Different Classifiers</h1>',unsafe_allow_html=True)

h1,h2 = st.columns([3,1])

with h1:
    st.title("Explore Different Classifiers")
    st.write("## Which one is the best:question:")

with h2:
    def load_animations(filepath:str):
        with open(filepath,'r',encoding="utf8") as f:
            return simplejson.load(f)

    icon = load_animations("icon.json")
    st_lottie(
        icon,
        speed=1,
        reverse=False,
        loop=True,
        quality='medium',
        height=None,
        width=None,
        key="icon"
    )


# st.write("")
# st.write("")


df_name = st.sidebar.selectbox("Select Dataset",["Iris","Breast Cancer","Wine Dataset"])


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

st.set_option('deprecation.showPyplotGlobalUse', False)


X, y = get_dataset(df_name)
st.sidebar.write(f"Shape of Dataset is **{X.shape}**")
st.sidebar.write(f"Number of Classes in the Dataset is **{len(np.unique(y))}**")

st.sidebar.write("")

#PLOT
st.sidebar.header(f"Plotting the {df_name} dataset using PCA")
st.sidebar.write("")
pca = PCA(2)   #2 is the number of dimensions
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha = 0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.sidebar.pyplot()

st.sidebar.write("")

classifier_name = st.sidebar.selectbox("Select Classifier",["KNN","SVM","Random Forest"])

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
acc = round(acc,4)
acc = acc*100;

# st.warning(f"## Selected Classifier :arrow_right: {classifier_name}")
st.header(f"{classifier_name} Results on {df_name} Dataset: ")


#PREDICTOR
# if df_name=="Iris":
#     input1 = st.number_input("Sepal Length (cm)", 3.0,10.0)
#     input2 = st.number_input("Sepal Width (cm)", 1.0,5.0)
#     input3 = st.number_input("Petal Length (cm)", 0.0,10.0)
#     input4 = st.number_input("Petal Width (cm)", 0.0,3.0)
    
#     prediction = classifier.predict(np.array(input1,input2,input3,input4))
#     st.write(prediction)


#PLOT CONFUSION MATRIX
st.header("Confusion Matrix")
st.write()
from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(y_test,y_pred)
fig2 = plt.figure
sns.heatmap(cm,annot=True)
st.pyplot()

#CLASSIFICATION REPORT
st.header("Classification Report")
report = classification_report(y_test,y_pred,output_dict=True)
st.dataframe(report,use_container_width=True)

st.info(f"Accuracy :arrow_right: {acc}%")

st.write("")
st.write("")

st.divider()

col1,col2,col3 = st.columns(3)
#how to use, about, etc
# Add "How to Use" Section
with col1:
    st.write("How to Use")
    st.caption("Choose a dataset from the sidebar using the 'Select Dataset' dropdown.")
    st.caption("Explore the dataset characteristics in the sidebar.")
    st.caption("Select a classifier from the 'Select Classifier' dropdown.")
    st.caption("Adjust the classifier parameters in the sidebar if necessary.")
    st.caption("See the classifier results, confusion matrix, and classification report.")

# Add "About" Section
with col2:
    st.write("About")
    st.caption("This Streamlit app allows you to explore different classifiers on three datasets: Iris, Breast Cancer, and Wine.")
    st.caption("It supports K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest classifiers.")
    st.caption("The application provides insights into the dataset through Principal Component Analysis (PCA) plots.")
    st.caption("Confusion matrix and classification report are displayed to evaluate classifier performance.")


# source code
with col3:
    st.write("Source Code")
    st.caption("The source code for this Streamlit app is available on GitHub.")
    st.caption("[Link to GitHub Repository](https://github.com/Devansh-Gupta-Official/classification-web-app)")

st.write("")
st.write("")

st.divider()

#ADDING FEEDBACK 1
# feedback = streamlit_feedback(
#     feedback_type="faces",
#     # optional_text_label="Please provide an explanation",
#     align="center"
# )

# if feedback == None:
#     pass
# elif feedback['score']=='😞':
#     st.text_input("Feedback",placeholder='Enter your Feedback')
# elif feedback['score']=='🙁':
#     st.text_input("Feedback",placeholder='Enter your Feedback')
# elif feedback['score']=='😐':
#     st.text_input("Feedback",placeholder='Enter your Feedback')
# elif feedback['score']=='🙂':
#     st.write("Thank you for your feedback!")
# elif feedback['score']=='😀':
#     st.write("Thank you for your feedback!")
#     st.balloons()


#ADDING FEEDBACK 2
def center_content():
    col1, col2, col3 = st.columns([3, 2, 1])
    return col1, col2, col3

stars = st_star_rating("Please rate your experience!", maxValue=5, defaultValue=0, key="rating", dark_theme=True)

# Centering content
col1, col2, col3 = center_content()

if stars in [1, 2, 3]:
    with col1:
        st.write("We value your feedback! Please let us know how we can improve.")
        feedback = st.text_area("Enter your Feedback", placeholder='Type here...', height=150)
        if st.button("Submit Feedback"):
            # Add logic to handle the feedback submission if needed
            st.success("Feedback submitted successfully!")

elif stars in [4, 5]:
    with col1:
        st.write("Thank you for your positive feedback!")
        st.balloons()
