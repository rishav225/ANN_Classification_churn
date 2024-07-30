import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten, Dense
import pickle
import numpy as np

desired_width=420
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',20)

# Load data set
data = pd.read_csv('/Users/rishavkumar/PycharmProjects/mylearning/ANN Classification/Churn_Modelling.csv')


# # Preprocess data
# Drop irrelevant data
data = data.drop(['RowNumber','CustomerId','Surname'], axis=1)

# Encode categorical data
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
# one hot encoder 'Geogropy'
onehot_geo=OneHotEncoder(sparse_output=False)
geo_onehot=onehot_geo.fit_transform(data[['Geography']])
geo_encoded_df=pd.DataFrame(geo_onehot, columns=onehot_geo.get_feature_names_out(['Geography']))

# Combine data
data=data.drop(['Geography'], axis=1)
data=pd.concat((data,geo_encoded_df), axis=1)

#Save the encoder and scaler file

with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)
with open('onehot_geo.pkl', 'wb') as file:
    pickle.dump(onehot_geo,file)
# Divide the data set into independent and dependent data set
X=data.drop(['Exited'], axis=1)
y=data['Exited']
# Split the data into training set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scale these feature
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print(X_train.shape[1])

model=Sequential()
model.add(Flatten(input_shape=(X_train.shape[1],))) ## HL 1
model.add(BatchNormalization())
model.add(Dense(300, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax')) ## Output layers
print(model.summary())
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=10)
history.model.save('model.h5')
print(history.history)