#python app.py

from flask import Flask,request,jsonify,render_template
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# prediction_endpoint="http://127.0.0.1:5000"

app = Flask(__name__)

@app.route("/",methods=["GET", "POST"])
def index():
   return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
   tfid_vectorizer=pickle.load(open(r"Models/tfid_vectorizer.pkl","rb"))
   predictor=pickle.load(open(r"Models/random_forest_classify.pkl","rb"))
   text_input=request.json["text"]
   try:
      if text_input:
         predicted_sentiment=do_prediction(tfid_vectorizer,predictor,text_input)
         return jsonify({"Prediciton":{predicted_sentiment}})
   except Exception as e:
      return jsonify({"ERROR":str(e)})
   
def do_prediction(tfid_vectorizer,predictor,text_input):
   corpus=[]
   lemmatizer=WordNetLemmatizer()
   review=re.sub('[^a-z A-Z 0-9]',' ',text_input)
   review=review.lower().split()
   review=[lemmatizer.lemmatize(y) for y in review if y not in stopwords.words('english')]
   review=" ".join(review)
   corpus.append(review)
   vectorizer=tfid_vectorizer.transform(corpus).toarray()
   prediction=predictor.predict(vectorizer)
   return "Positive" if prediction[0]==1 else "Negative"
   
if __name__=="__main__":
   app.run(port=5000,debug=True)