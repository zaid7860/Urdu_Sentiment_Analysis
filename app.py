from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
 
def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("bt", "but").replace("MASHA'ALLAH", "mashallah").replace("INSHA'ALLAH", "inshallah")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x = re.sub(r"http\S+", "", x)
    
    porter = PorterStemmer()
    pattern = re.compile(r'\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    
    return x
    
   
###################################################

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('vector.pkl')
    to_predict_list = request.form.to_dict()
    review_text = preprocess(to_predict_list['review_text'])
    pred = clf.predict(count_vect.transform([review_text]))
    if pred==0:
        prediction = "Negative"
    elif pred==1:
        prediction = "Positive"
        return prediction
    elif pred==3:
        prediction = "Neutral"
    return prediction

    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
