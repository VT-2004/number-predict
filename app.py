import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import io
from dotenv import load_dotenv # Import load_dotenv

# --- Load Environment Variables ---
load_dotenv() # This line loads variables from a .env file if it exists

# --- Flask App Setup ---
app = Flask(__name__)

# Get the allowed origin from environment variables
# If FRONTEND_URL is not set, it defaults to a common development port or a placeholder
allowed_frontend_urls_str = os.getenv('FRONTEND_URL', 'http://localhost:5500,http://127.0.0.1:5500')
allowed_frontend_urls = [url.strip() for url in allowed_frontend_urls_str.split(',')]

CORS(app, resources={r"/predict/*": {"origins": allowed_frontend_urls}})
# If using a list of URLs:
# CORS(app, resources={r"/predict/*": {"origins": allowed_frontend_urls}})


# --- Model Loading ---
# We load the model here to ensure it's loaded only once.
# This makes subsequent prediction requests much faster.
try:
    model_path = os.path.join(os.getcwd(), 'mnist_model.keras')
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    # If the model fails to load, we log the error and set the model to None.
    # The /predict route will handle this case gracefully.
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    # Gracefully handle the case where the model failed to load at startup.
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data_base64 = data['image'].split(',')[1]
        
        # We need to make sure we process the image exactly as the model expects.
        image_bytes = base64.b64decode(image_data_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale ('L' mode) and resize to 28x28 pixels
        image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert the image to a NumPy array, invert the colors (since MNIST is white-on-black)
        # and normalize the pixel values to be between 0 and 1.
        image_array = np.array(image, dtype=np.float32)
        image_array = 255 - image_array
        image_array /= 255.0
        
        # The model expects a batch of images, so we add a batch dimension.
        # It also expects the channel dimension.
        preprocessed_image = np.expand_dims(image_array, axis=0)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)

        # Make a prediction
        prediction = model.predict(preprocessed_image, verbose=0)
        predicted_label = np.argmax(prediction[0])

        return jsonify({'prediction': int(predicted_label)})
    
    except Exception as e:
        # Catch any other errors during prediction and return a 500 status code.
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

# --- Main Entry Point ---
if __name__ == '__main__':
    # You can change the port here if needed.
    # When deployed on Render, this part of the code is not used.
    app.run(debug=True, port=5000)