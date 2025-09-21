import os
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# --- Flask App Setup ---
app = Flask(__name__)
# Enable CORS to allow the frontend (running on a different port) to access the backend.
CORS(app)

# --- Model Loading ---
# This loads the Keras model when the server starts.
# We load it here once to avoid reloading it for every request, which would be slow.
# The 'my_number_classifier.keras' file must be in the same directory as this script.
try:
    model_path = os.path.join(os.getcwd(), 'mnist_model.keras')
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    data = request.get_json(force=True)
    image_data_base64 = data['image'].split(',')[1]

    try:
        # --- Add this print statement to check if the image data is being received ---
        print("Received image data. Length:", len(image_data_base64))
        
        image_bytes = base64.b64decode(image_data_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure the image has no alpha channel before converting to grayscale
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        image_array = np.array(image, dtype=np.float32)

        image_array = 255 - image_array
        image_array /= 255.0

        # --- Add this print statement to check the preprocessed image shape ---
        print("Preprocessed image shape:", image_array.shape)

        preprocessed_image = np.expand_dims(image_array, axis=0)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)

        prediction = model.predict(preprocessed_image, verbose=0)
        predicted_label = np.argmax(prediction[0])

        return jsonify({'prediction': int(predicted_label)})
    
    except Exception as e:
        # --- This will print the actual error traceback to the console ---
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

# --- Main Entry Point ---
if __name__ == '__main__':
    # You can change the port here if needed.
    app.run(debug=True, port=5000)
