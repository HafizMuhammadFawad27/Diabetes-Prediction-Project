# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st
# Load the dataset
data = pd.read_csv('diabetes.csv')  # Make sure the file path is correct
# Display first 5 rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Dataset ki basic information
print(data.info())

# Dataset ka description
print(data.describe())
# Features (X) aur Target (y) mein divide karna
X = data.drop('Outcome', axis=1)  # Features (sab columns except 'Outcome')
y = data['Outcome']  # Target (Outcome column)

# Data ko train aur test sets mein divide karna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data ko standardize karna (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Support Vector Machine (SVM) model banayein
model = SVC(kernel='linear')

# Model ko train karein
model.fit(X_train, y_train)
# Predictions karein
y_pred = model.predict(X_test)

# Model ki accuracy check karein
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Streamlit app
st.title('Diabetes Prediction App')

# User input lena
st.sidebar.header('User Input Features')

# Function to get user input
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    # Dictionary banayein
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    
    # DataFrame mein convert karein
    features = pd.DataFrame(data, index=[0])
    return features

# User input ko collect karein
input_df = user_input_features()

# Display user input
st.subheader('User Input:')
st.write(input_df)

# Prediction karein
prediction = model.predict(scaler.transform(input_df))

# Result display karein
st.subheader('Prediction:')
if prediction[0] == 1:
    st.write('The person has Diabetes.')
else:
    st.write('The person does not have Diabetes.')
    