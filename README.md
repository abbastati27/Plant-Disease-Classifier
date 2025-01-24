# Plant Disease Classification
This project is a web application that uses machine learning models to classify plant diseases based on images uploaded by users. It uses Flask for the backend, TensorFlow for the machine learning model, and Keras for model prediction.


# Features
1. Upload an Image: Users can upload images of plants to the web interface.
Plant Disease Classification: The uploaded image is processed, and the model predicts the disease if present, based on a pre-trained model.
2. Confidence Score: The model returns a confidence score for the classification, indicating how certain the model is about the prediction.


# Technologies Used
1. Flask: A lightweight Python web framework for creating web applications.
2. TensorFlow/Keras: A deep learning framework used to train and load the model for plant disease classification.
3. NumPy: Used for handling the array manipulations required for image processing.
4. HTML/CSS: For the frontend interface of the web application.
5. Python 3.8+: Required for running the application.


# Requirements
To run this project locally, you'll need to install the following dependencies:

1. Flask
2. TensorFlow
3. Keras
4. NumPy
5. Pillow (for image handling)


# You can install all required dependencies using pip:-
pip install -r requirements.txt
Note: You may need to set up a virtual environment for Python.


# Run the Flask Application:-
python app.py


# Model Details
The model is a Convolutional Neural Network (CNN) that has been trained on a plant disease dataset. It classifies plants into the following categories (you may need to adjust these based on your dataset):
1. Potato___Early_blight
2. Potato___Late_blight
3. Potato___Healthy
The model expects input images of size 256x256 pixels and uses a softmax activation function in the final layer to classify the image into one of the disease categories.


# API Endpoints
/predict (POST)
1. Description: This endpoint accepts an image file via a POST request and returns a prediction of the plant disease.
2. Parameters:
file (required): The image file to be classified.
3. Response: A JSON object containing:
prediction: The predicted disease class (e.g., Potato___Early_blight).
confidence: The confidence score of the prediction (percentage).


# Troubleshooting
1. Error - Input Shape Mismatch: If you get a ValueError related to image input shapes, ensure that your model expects images of size 256x256 and  update the image resizing accordingly.
2. Model Not Found: Make sure the model file (plant_disease_model_2.keras) is in the correct directory and is loaded properly in the application.


# License
This project is licensed under the MIT License - see the LICENSE file for details.