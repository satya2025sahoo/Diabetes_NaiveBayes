from flask import Flask,request,app,render_template
from flask import Response
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

scaler=joblib.load('/config/workspace/Model/standardScalar.pkl')
model=joblib.load('/config/workspace/Model/gmodel_BNB.pkl')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict():
    result=''

    if request.method=='POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))

        data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        y_pred=model.predict(data)

        if y_pred[0]==1:
            result='DIABETIC'
        else:
            result='NON DIABETIC'

        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html') 

if __name__=="__main__":
    app.run(host="0.0.0.0")
