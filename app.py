import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import numpy as np
desired_width=420
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the pickel file
model = tf.keras.models.load_model('/Users/rishavkumar/PycharmProjects/mylearning/ANN Classification/model.h5')

#load encoder pickel file

with open('/Users/rishavkumar/PycharmProjects/mylearning/ANN Classification/label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('/Users/rishavkumar/PycharmProjects/mylearning/ANN Classification/onehot_geo.pkl', 'rb') as f:
    onehot_geo = pickle.load(f)

with open('/Users/rishavkumar/PycharmProjects/mylearning/ANN Classification/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## stremlit app
st.title("Custumer churn prediction")

# user inputs

geo = st.selectbox('Geography', onehot_geo.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input("Credit score")
estimated_salary  = st.number_input('Estimated salary')
tenure = st.slider('Tenure', 0, 10)
num_of_product = st.slider('Number of product', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare the input data

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([Gender])[0]],
    'Age': [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_product],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# one hot geography value
geo_encoded = onehot_geo.transform([[geo]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the input
input_data_scaled  = scaler.transform(input_data)

## Find predection
prediction = model.predict(input_data_scaled)

prediction_probab = prediction[0][0]

if prediction_probab > 0.5:
    st.write('The custumer is likely to churn.')
else:
    st.write('The custumer is not likely to churn.')