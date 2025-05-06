from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import groq
import os
from flask import jsonify
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load the trained model and scaler
basedir = os.path.abspath(os.path.dirname(__file__))
model = joblib.load(os.path.join(basedir, 'crop_app.pkl'))
scaler = joblib.load(os.path.join(basedir, 'scaler.pkl'))

# Load rainfall model and encoders
rain_model = joblib.load(os.path.join(basedir, 'rainfall_model.pkl'))
month_encoder = joblib.load(os.path.join(basedir, 'month_encoder.pkl'))
district_encoder = joblib.load(os.path.join(basedir, 'district_encoder.pkl'))
rain_scaler = joblib.load(os.path.join(basedir, 'rainfall_scaler.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the soil parameters from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare the input for the model
        arr = [[N, P, K, temperature, humidity, ph, rainfall]]
        arr_scaled = scaler.transform(arr)

        # Get the crop prediction
        prediction = model.predict(arr_scaled)

        # Get the AI insights for the predicted crop
        crop_info = get_crop_info_from_groq(prediction[0])

        # Pass the prediction and crop info to the result page
        return render_template('result.html', prediction=prediction[0], ai_summary=crop_info)
    
    return render_template('predict.html')

@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    data = request.get_json()
    district = data.get('district')
    month = data.get('month')

    if not district or not month:
        return jsonify({'error': 'Both district and month are required'}), 400

    try:
        # Encode and scale inputs
        district_encoded = district_encoder.transform([district])[0]
        month_encoded = month_encoder.transform([month])[0]
        X_input = np.array([[district_encoded, month_encoded]])
        X_scaled = rain_scaler.transform(X_input)

        # Predict rainfall
        predicted_rainfall = rain_model.predict(X_scaled)[0]
        return jsonify({'rainfall': round(predicted_rainfall, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # app.run(debug=True)
    pass
