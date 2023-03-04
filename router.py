import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template,request
from heardDisease_web import preprocess_new

## intiliaza
app = Flask(__name__)

model = joblib.load('logisticRg_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])
        
        X_new = (([age],[sex],[cp],[trestbps],[chol],[fbs],[restecg],[thalach],[exang],[oldpeak],[slope],[ca],[thal]))
        
        predict = preprocess_new(X_new)
                
        return render_template('index.html', y_predict = predict)
    else:
        return render_template('index.html')
## termianl
if __name__ == '__main__':
    app.run(debug=True)