from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import os

app=Flask(__name__)
cors=CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Attempt to load the trained model, but do not crash the server if it fails
model = None
model_load_error = None
model_path = os.path.join(BASE_DIR, 'LinearRegressionModel.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as exc:
    model_load_error = str(exc)
    print(f"Failed to load model from {model_path}: {model_load_error}")

car=pd.read_csv(os.path.join(BASE_DIR, 'Cleaned_Car_data.csv'))

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    if model is None:
        return (f"Model not loaded: {model_load_error}. Please retrain and save the model with the current scikit-learn version, or install the original version used to create the pickle.", 500)

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    # Coerce numeric fields and validate
    try:
        year = int(year)
    except Exception:
        return ("Invalid year value", 400)

    try:
        driven = int(driven)
    except Exception:
        return ("Invalid kms driven value", 400)

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True, use_reloader=False)
