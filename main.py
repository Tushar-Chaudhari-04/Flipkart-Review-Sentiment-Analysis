#streamlit run main.py

import requests
import streamlit as st

prediction_endpoint="http://127.0.0.1:5000/predict"

st.title("Sentiment Analysis of reviews ...")
user_input=st.text_input("Enter text and click on predict","")

# Headers (optional)
headers = {
    "Content-Type": "application/json", 
}

if st.button("Predict"):
   response=requests.post(
   url=prediction_endpoint,
   json={"text":user_input},
   headers=headers
  )
   
   data = response.text
   st.write(f"Predicted sentiment: {data}")

  

