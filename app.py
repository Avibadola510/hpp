import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Feature names (MUST match training order)
feature_names = [
    'MedInc',
    'HouseAge',
    'AveRooms',
    'AveBedrms',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]

@app.route('/')
def home():
    return render_template('home.html')


# ------------------- API ROUTE -------------------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    # Convert JSON input into DataFrame with correct feature order
    input_df = pd.DataFrame([data], columns=feature_names)
    
    # Scale input
    scaled_data = scaler.transform(input_df)
    
    # Predict
    output = regmodel.predict(scaled_data)
    
    return jsonify(float(output[0]))


# ------------------- FORM ROUTE -------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    
    # Convert to DataFrame with correct feature names
    input_df = pd.DataFrame([data], columns=feature_names)
    
    # Scale input
    scaled_data = scaler.transform(input_df)
    
    # Predict
    output = regmodel.predict(scaled_data)[0]
    
    return render_template(
        "home.html",
        prediction_text="The House price prediction is {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)