import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('decision_model.pkl', 'rb'))

dataset= pd.read_csv('PCA and NN Dataset3.csv')
X = dataset.iloc[:, 1:10].values
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 1:10]) 
#Replacing missing data with the calculated mean value  
X[:, 1:10]= imputer.transform(X[:, 1:10])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary):
  output= model.predict(sc.transform([[CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary]]))
  print("Output", output)
  if output==[0]:
    prediction="customer not left the bank"
  else:
    prediction="customer left the bank"
  print(prediction)
  return prediction
def main():
    st.title("customer left the bank prediction")
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
  
    CreditScore =  st.text_input("CreditScore","Type Here")
    Geography =  st.text_input("Geography","Type Here")
    Gender = st.text_input("Gender","Type Here")
    Age =  st.text_input("Age","Type Here")
    Tenure = st.text_input("Tenure","Type Here")
    Balance =  st.text_input("Balance","Type Here")
    HasCrCard =  st.text_input("HasCrcard","Type Here")
    IsActiveMember = st.text_input("IsActiveMember","Type Here")
    EstimatedSalary =  st.text_input("EstimatedSalary","Type Here")
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.text("Developed by Pritesh Kumar")
      st.text("Student , Department of Computer Engineering")

if __name__=='__main__':
  main()
   
