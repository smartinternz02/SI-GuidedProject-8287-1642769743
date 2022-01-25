import numpy as np 
import pandas as pd 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('CKD.pkl','rb'))

@app.route('/')
def home():
    return render_template('ckd.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['red_blood_cells','pus_cell','diabetesmellitus','pedal_edema','anemia','coronary_artery_disease','blood_glucose_random','blood_urea']

    df = pd.DataFrame(features_value, columns=features_name)

    output = model.predict(df)
    print(output)
   

    if output==1:
         return render_template('ckd.html',prediction_text='You have CHRONIC KIDNEY DISEASE')
    else:
        return render_template('ckd.html',prediction_text='You do not have CHRONIC KIDNEY DISEASE')

if __name__ == '__main__':
    app.run(debug=True)