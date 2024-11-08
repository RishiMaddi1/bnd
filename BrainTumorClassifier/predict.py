from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import os
import base64

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model for brain tumor detection
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Define labels for brain tumor classification
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
image_size = 150

# Variable to store results
results_storage = []

def predict_image(image_path):
    # Preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img_array = np.array(img).reshape(1, image_size, image_size, 3)

    # Predict and return the result
    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join('temp', file.filename)
    file.save(temp_file_path)

    # Make a prediction
    predicted_label = predict_image(temp_file_path)

    # Optionally, remove the temporary file after prediction
    os.remove(temp_file_path)

    return jsonify({'predicted_label': predicted_label})

# New route to handle Base64 image and combined predictions
@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    data = request.get_json()
    if 'type' not in data or 'image' not in data:
        return jsonify({'error': 'Missing type or image'}), 400

    prediction_result = ""
    
    if data['type'] == 'brain_tumor':
        # Handle Brain Tumor Detection
        image_data = base64.b64decode(data['image'])
        temp_file_path = 'temp/temp_image.jpg'
        
        with open(temp_file_path, 'wb') as f:
            f.write(image_data)

        predicted_label = predict_image(temp_file_path)
        os.remove(temp_file_path)

        # Store the result
        patient_history = data.get('medical_history', '')
        prediction_result = f"Predicted Tumor Type: {predicted_label}\nMedical History: {patient_history}"
        
        # Store the result in the global variable
        results_storage.append(prediction_result)  # Store the result for later use

    elif data['type'] == 'diabetes':
        # Handle Diabetes Prediction (placeholder for future implementation)
        blood_sugar = data.get('blood_sugar')
        insulin = data.get('insulin')
        bmi = data.get('bmi')
        age = data.get('age')
        symptoms = data.get('symptoms', '')

        # Placeholder for diabetes prediction logic
        prediction_result = f"Diabetes Prediction based on inputs: {blood_sugar}, {insulin}, {bmi}, {age}. Symptoms: {symptoms}"

    else:
        return jsonify({'error': 'Invalid type'}), 400

    return jsonify({'result': prediction_result})

if __name__ == "__main__":
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    app.run(host='0.0.0.0', port=5000)