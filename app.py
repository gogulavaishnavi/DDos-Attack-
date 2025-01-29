import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('C:/Users/Vaishnavi/Desktop/PD/final_model.pkl')# Change to your model file name
scaler = joblib.load('C:/Users/Vaishnavi/Desktop/PD/scaler.pkl')  # Change to your scaler file name

@app.route('/')
def home():
    return "Model Deployment is working!"

# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # Assuming data contains features in the same order as training data
        input_data = np.array(data['input_features']).reshape(1, -1)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict using the loaded model
        prediction = model.predict(input_data_scaled)
        
        # Return the result as a JSON response
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Start the Flask application
    app.run(debug=True)
