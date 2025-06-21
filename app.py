import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
import numpy as np
import streamlit as st

#Loading the model, the encoders and the scalers
model= tf.keras.models.load_model('model.h5')

#Loading the encoders and the scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_encoder_Geo.pkl', 'rb') as file:
    one_hot_encoder_Geo=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

#Streamlit App
st.title('Customer Churn Prediction')

#User Input
name=st.text_input('Name of Customer', placeholder='You name here')
gender=st.selectbox('Gender',label_encoder_gender.classes_)
geography=st.selectbox('Geography', one_hot_encoder_Geo.categories_[0])
age=st.slider('Age', 18,90)
col1,col2,col3=st.columns(3)
with col1:
    balance=st.number_input('Bank Balance')
with col2:
    credit_score=st.number_input('Credit Score')
with col3:
    estimated_salary=st.number_input('Estimated Salary')
col4,col5=st.columns(2)
with col4:
    num_of_products=st.slider("Number of Products")
with col5:
    tenure=st.slider('Tenure',0,10)
col6,col7=st.columns(2)
with col6:
    has_cr_card=st.selectbox('Has Credit Card', [0,1])
with col7:
    is_active_member=st.selectbox('Is Active Member', [0,1])
if name.strip() == "":
    button = st.button("Submit", disabled=True)
else:
    button = st.button("Submit")

#prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoder=one_hot_encoder_Geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoder, columns=one_hot_encoder_Geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the input data
input_scaled=scaler.transform(input_data)

#predict churn
prediction=model.predict(input_scaled)
prediction_probability=prediction[0][0]

if button==True:
    if prediction_probability>0.5:
        st.write("Customer likely to stay")
    else:
        st.write('Customer Likely to leave')
        st.write(f'Predicted Churn Probability:{prediction_probability:.2f}')