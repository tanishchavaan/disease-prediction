import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import webbrowser


model_filename = 'disease_prediction_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)


data = pd.read_csv('Training.csv')
data = data.dropna(axis=1)
symptoms = data.columns[:-1]  # All columns excluding diseases column


encoder = LabelEncoder()

encoder.fit(data['prognosis'])

def google_search(symptoms):
    query = f"treatment%20for%20{symptoms.replace(' ', '%20')}"  # Encoding space with %20
    url = f"https://www.google.com/search?q={query}"
    st.markdown(f"[Search Treatment for {symptoms}]({url})", unsafe_allow_html=True)




st.title("Medical Assistant")


st.sidebar.title("Additional Features")


st.markdown("<h2>Disease Prediction Based on Symptoms</h2>", unsafe_allow_html=True)


selected_symptoms = [st.selectbox(f'Select symptom {i+1}',  ['None']+list(symptoms) ) for i in range(5)]

count = 0

input_data = np.zeros(len(symptoms))
for symptom in selected_symptoms:
    if symptom != 'None':
        input_data[list(symptoms).index(symptom)] = 1
    if symptom == 'None' :
        count = count +1 ;

if(count < 5) :
    if st.button('Predict Disease' , key = 'predict_button'):
        prediction = model.predict([input_data])
        #prob = model.predict_proba([input_data])[0]
        disease = encoder.inverse_transform(prediction)[0]
        st.success(f'The predicted disease is: **{disease}**')
        #st.success(f'The predicted disease probability is: **{max(prob)}**')



if(count<5) :    
    if st.button("Search Treatment"):
        prediction = model.predict([input_data])
        Disease = encoder.inverse_transform(prediction)[0]
        google_search(Disease)


st.write("*Disclaimer: Predictions from the model might not always be accurate. Consult a doctor for professional medical advice.*")
