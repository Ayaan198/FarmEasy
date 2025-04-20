from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from groq_ai import get_crop_info_from_groq
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load the trained model and scaler
basedir = os.path.abspath(os.path.dirname(__file__))
model = joblib.load(os.path.join(basedir, 'crop_app.pkl'))
scaler = joblib.load(os.path.join(basedir, 'scaler.pkl'))

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

if __name__ == '__main__':
    app.run(debug=True)
