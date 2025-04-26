from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "https://image-classifier-frontend.vercel.app"}})

# Load the pickled model
model = joblib.load('cnn_model.pkl')

# Class names (from CIFAR-10 dataset)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
import io

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file as bytes
        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(32, 32))  # Use BytesIO to handle the file as a byte stream
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]

        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
