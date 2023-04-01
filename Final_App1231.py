import pandas as pd
import pickle
import shap
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact,fixed,interactive,interact_manual
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import numpy as np
import xgboost as xgb
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from PIL import Image
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# loading in the model to predict on the data

#reg=pickle.load(open("xgb_reg.pkl","rb"))


def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs

# Pre-processing user input


def first_row_to_dict(X_train):
    first_row = X_train.iloc[0]
    return {col_name: first_row[col_name] for col_name in X_train.columns}

def predict(data):
    # Convert the input data to a DMatrix
    dmatrix = xgb.DMatrix(data)
    # Use the loaded model to make predictions
    return regressor.predict(dmatrix)
def prediction(df1):           
        
    
    X_train = pd.read_csv('X_train.csv')

    y_train = pd.read_csv('y_train.csv')

    X_test = df1

    y_test = pd.read_csv('y_test.csv')

    

    model = xgb.Booster()
    model.load_model('model.bin')
    output = first_row_to_dict(X_test)
    x_test = pd.DataFrame(output)
    #x_test = pd.DataFrame(output, index=[0])
    # pass this data to model
    #prediction  = predict(x_test)
    prediction  = model.predict(xgb.DMatrix(x_test))
    x_test["Demand_Qty"]=list(prediction)
    x_test["Location"]="Atlanta"
    x_test["Period_of_time"]="1-Week-Out"
    col=["Location","Material","Period_of_time","Demand_Qty",]
    predictions=x_test[col]
    explainer = shap.Explainer(model, X_train)


    shap_values = explainer(X_test)
    return predictions,shap_values,X_test

        


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Demand Sensing Model")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """    
    <h3 style ="color:white;">Demand Prediction Prediction </h3>
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
      
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
      df = pd.read_csv(uploaded_file)
      #df=df.head(2)
      st.write(df.head(5))
    
    
    
    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result,r1,r2 = prediction(df) 
        
        
   
        r=pd.DataFrame(result)
        
        st.write(r)
        st.title("Strength of Signals on  Model")
        def st_shap(shap_values, features):
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, features, plot_type="bar", show=False)
            ax.set_xlabel("Signal strength")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
        #st_shap(shap.summary_plot(r1,r2, plot_type="bar"))
        st_shap(r1,r2)
        
        

if __name__ == '__main__':
    main()