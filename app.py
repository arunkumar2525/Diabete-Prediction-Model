
# Importing API Libraries like flask to deploy diabete model in production
from flask import Flask,request,render_template
import numpy as np
import pickle

# load the pickle file Random Forest Classifier
fileName="Diabete_Model_Prediction.pkl"
Diabete_Classifier=pickle.load(open(fileName,'rb'))

# Creating the object of the api constructor
app=Flask(__name__)

# Calling the default HomePage Pages Opens
@app.route('/')
def homePage():
    return render_template('HomePage.html')

@app.route('/predict',method=['POST'])
def model_prediction():
    if request.method=='POST':
        preg=int(request.form['pregnancies'])
        glucose=int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        
        data=np.array([[preg,glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction=Diabete_Classifier.predict(data)
        return render_template('ResultPage.html',prediction=my_prediction)

if __name__=="__main__":
    app.run(debug=True)
    