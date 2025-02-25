from matplotlib import scale
import numpy as np
import joblib
import streamlit as st

model=joblib.load('model_scaled.pkl')
scaled=joblib.load('scaled.pkl')

st.title("machine learning model deployment")
st.write("enter your medical detials to know about your diabetes status")


st.sidebar.header("your medical records")
preg=st.sidebar.number_input("preg",min_value=0,max_value=100, value=50,step=0.1)
plas=st.sidebar.number_input("plas",min_value=0,max_value=100, value=50,step=0.1)
pres=st.sidebar.number_input("pres",min_value=0,max_value=100, value=50,step=0.1)
skin=st.sidebar.number_input("skin",min_value=0,max_value=100, value=50,step=0.1)
test=st.sidebar.number_input("test",min_value=0,max_value=100, value=50,step=0.1)
mass=st.sidebar.number_input("mass",min_value=0,max_value=100, value=50,step=0.1)
pedi=st.sidebar.number_input("pedi",min_value=0,max_value=100, value=50,step=0.1)
age=st.sidebar.number_input("age",min_value=0,max_value=100, value=50,step=0.1)

input_data=np.array([[preg,plas,pres,skin,test,mass,pedi,age]])
scaled_input=scale.transform(input_data)

if st.button("predict"):
    prediction=model.predict(scaled_input)
    st.success(f"Prediction:{prediction[0]}")
    




    