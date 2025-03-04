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
st.title('Diabetes Prediction Files')

# Define standard/normal values for comparison
standard_values = {
    'Pregnancies': 3,
    'Glucose': 117,
    'BloodPressure': 72,
    'SkinThickness': 23,
    'Insulin': 30,
    'BMI': 32.0,
    'DiabetesPedigreeFunction': 0.3725,
    'Age': 29
}

# Sidebar for user input (sliders)
st.sidebar.header('User Values')

# Function to get user input from sliders
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, standard_values['Pregnancies'])
    glucose = st.sidebar.slider('Glucose', 0, 200, standard_values['Glucose'])
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, standard_values['BloodPressure'])
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, standard_values['SkinThickness'])
    insulin = st.sidebar.slider('Insulin', 0, 846, standard_values['Insulin'])
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, standard_values['BMI'])
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, standard_values['DiabetesPedigreeFunction'])
    age = st.sidebar.slider('Age', 21, 81, standard_values['Age'])
    
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

# Main section for file upload
st.header('_______________________________________')

# File uploader for CSV or Excel files
uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = pd.read_excel(uploaded_file)
    
    # Display the uploaded data
    st.subheader('Uploaded File:')
    st.write(input_df)
    
    # Ensure the uploaded data has the same columns as the training data
    required_columns = X.columns
    if all(column in input_df.columns for column in required_columns):
        # Create a DataFrame for standard values
        standard_df = pd.DataFrame([standard_values])
        
        # Display standard values
        st.subheader('Normal Values:')
        st.write(standard_df)
        
        # Standardize the uploaded data
        input_scaled = scaler.transform(input_df[required_columns])
        
        # Make predictions
        predictions = model.predict(input_scaled)
        
        # Add predictions to the DataFrame
        input_df['Prediction'] = predictions
        
        # Display the predictions
        st.subheader('Predictions:')
        for i, row in input_df.iterrows():
            st.write(f"Patient {i+1}:")
            st.write(row)
            if row['Prediction'] == 1:
                st.write("**This person has Diabetes.**")
            else:
                st.write("**This person does not have Diabetes.**")
            st.write("---")
    else:
        st.error(f"Uploaded file must contain the following columns: {required_columns}")

# Slider section (independent of file upload)
st.header('Diabetes Prediction Values')

# User input ko collect karein
input_df = user_input_features()

# Display user input
st.subheader('Patient Values:')
st.write(input_df)

# Display standard values
st.subheader('Normal Values:')
standard_df = pd.DataFrame([standard_values])
st.write(standard_df)

# Prediction karein
prediction = model.predict(scaler.transform(input_df))

# Result display karein
st.subheader('Prediction:')
if prediction[0] == 1:
    st.write('**This person has Diabetes.**')
else:
    st.write('**This person does not have Diabetes.**')
