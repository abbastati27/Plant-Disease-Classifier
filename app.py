from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from io import BytesIO
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array


app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("plant_disease_model_2.keras")
print("Model loaded successfully")

# Load class names (ensure you load them from your training setup if necessary)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']  # Replace with your actual class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from the form
        file = request.files['file']

        # Convert the file to a BytesIO object
        img_bytes = file.read()

        # Preprocess the image like in the Jupyter notebook
        img = load_img(BytesIO(img_bytes), target_size=(256, 256))  # Adjust size as per your model

        # Convert to an array using img_to_array from tensorflow.keras.utils
        img_array = img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        predictions = model.predict(img_array)

        # Get the predicted class index and confidence score
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions) * 100  # Convert to percentage

        # Extract the class name
        predicted_class_name = class_names[predicted_class_index].split('___')[1]
        predicted_class_name = predicted_class_name.replace('_', ' ')

        # Prepare the response with prediction and confidence
        response = {
            'prediction': predicted_class_name,
            'confidence': str(round(float(confidence_score), 2))
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
