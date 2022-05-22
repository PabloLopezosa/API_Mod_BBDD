from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pickle
import os
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from resources import dict_emotions, signs_texts, remove_stopwords, spanish_stemmer

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

model = pickle.load(open('finished_model.model','rb'))

@app.route('/', methods=['GET'])
def home():
    return "MODELO MACHINE LEARNING CHATBOT ENLACE"

@app.route('/api/v1/consulta', methods=['GET'])
def consulta():

    consulta = request.args.get('consulta', None)


    consulta = signs_texts(consulta)
    consulta = remove_stopwords(consulta)
    consulta = spanish_stemmer(consulta)

    if consulta is None :
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict(pd.Series(consulta))[0]
    
    return jsonify({'Respuesta de ENLACE' : dict_emotions[prediction]})


#http://127.0.0.1:5000/api/v1/consulta?consulta=%20economico

app.run()