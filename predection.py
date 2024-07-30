import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import numpy as np
desired_width=420
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)

# Load the pickel file
model=load_model('/Users/rishavkumar/PycharmProjects/mylearning/ANN Classification/model.h5')

#load encoder pickel file

with open('/ANN Classification/label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('/ANN Classification/onehot_geo.pkl', 'rb') as f:
    onehot_geo = pickle.load(f)

with open('/ANN Classification/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_df = pd.read_csv('/ANN Classification/predict_input.csv')
print(input_df)

#one-hot encoder
geo_encoder = onehot_geo.transform([input_df['Geography']])
geo_encoder_df = pd.DataFrame(geo_encoder, columns=onehot_geo.get_feature_names_out(['Geography']))
print(geo_encoder_df)

## Combile one hot with input data
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoder_df], axis=1)
print(input_df)

## Encode categeorical variables
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
input_df=input_df.drop(['Geography'], axis=1)
print(input_df)

# scale the input
input_scaled=scaler.transform(input_df)

## Find predection
prediction = model.predict(input_scaled)

prediction_probab = prediction[0][0]

print(prediction_probab)

print(label_encoder_gender.classes_)