from flask import Flask, request, jsonify, render_template
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_wtf.csrf import CSRFProtect
from flask import send_from_directory


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
csrf = CSRFProtect(app)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="skin_lesion_model.tflite")
interpreter.allocate_tensors()

# Assuming input_data is a dictionary with a field 'image_data' representing the image pixel values
def preprocess_data(input_data):
    # Convert the image pixel values to a NumPy array
    img_array = np.array(input_data, dtype=np.float32)  # Ensure FLOAT32 data type

    # Resize the image to match your model's input shape
    img_array = cv2.resize(img_array, (32, 32))

    # Add an extra dimension to match the input shape of your model
    input_tensor = np.expand_dims(img_array, axis=0)  # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=-1)  # Add channel dimension
    input_tensor = np.concatenate([input_tensor] * 3, axis=-1)  # Repeat the single channel to match three channels
    print("Input tensor shape:", input_tensor.shape)

    return input_tensor


def predict_from_frame(frame):
    # Perform inference
    input_data = preprocess_data(frame)

    # Get the input tensor index
    input_tensor_index = interpreter.get_input_details()[0]['index']

    # Set the input tensor using the input tensor index
    interpreter.set_tensor(input_tensor_index, input_data.astype(np.float32))  # Convert to FLOAT32
    interpreter.invoke()

    # Get the output tensor
    output_tensor_index = interpreter.get_output_details()[0]['index']
    output_data = interpreter.get_tensor(output_tensor_index)

    # Post-process the output if needed
    return output_data.tolist()

@app.route('/')
def home():
    return render_template('startup.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the JSON data from the request
    data = request.get_json()

    # Preprocess the input data
    input_data = preprocess_data(data['image_data'])

    # Get the input tensor index
    input_tensor_index = interpreter.get_input_details()[0]['index']

    # Set the input tensor using the global interpreter
    interpreter.set_tensor(input_tensor_index, input_data.astype(np.float32))  # Convert to FLOAT32

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_tensor_index = interpreter.get_output_details()[0]['index']
    output_data = interpreter.get_tensor(output_tensor_index)

    # Process the output data
    result = process_output(output_data)

    # Return the prediction
    return jsonify({'prediction': result})

def process_output(output_data):
    # Assuming output_data is a list of probabilities for each class
    predicted_class_index = np.argmax(output_data)

    # Check if the predicted class index is within the valid range
    if 0 <= predicted_class_index < len(lesion_type_dict):
        predicted_class_key = list(lesion_type_dict.keys())[predicted_class_index]
        predicted_class_label = lesion_type_dict[predicted_class_key]

        return {'prediction': predicted_class_label}
    else:
        return {'prediction': 'Unknown Class'}

# New route for webcam predictions
@app.route('/predict_webcam', methods=['GET', 'POST'])
def predict_webcam():
    if request.method == 'POST':
        data = request.get_json()

        # Preprocess the input data
        input_data = preprocess_data(data['image_data'])

        # Get the input tensor index
        input_tensor_index = interpreter.get_input_details()[0]['index']

        # Set the input tensor using the global interpreter
        interpreter.set_tensor(input_tensor_index, input_data.astype(np.float32))  # Convert to FLOAT32

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_tensor_index = interpreter.get_output_details()[0]['index']
        output_data = interpreter.get_tensor(output_tensor_index)

        # Process the output data
        result = process_output(output_data)

        # Return the prediction
        return jsonify(result)

    return render_template('index.html')

def process_output(output_data):
    # Assuming output_data is a list of probabilities for each class
    predicted_class_index = np.argmax(output_data)

    # Debugging prints
    print('Output Data:', output_data)
    print('Predicted Class Index:', predicted_class_index)

    predicted_class_key = list(lesion_type_dict.keys())[predicted_class_index]
    predicted_class_label = lesion_type_dict.get(predicted_class_key, 'Unknown Class')

    # Debugging print
    print('Predicted Class Key:', predicted_class_key)

    return {'prediction': predicted_class_label}



# Load TensorFlow Lite binary model
binary_interpreter = tf.lite.Interpreter(model_path="skin_lesion_binary_model.tflite")
binary_interpreter.allocate_tensors()

# New route for binary predictions
@app.route('/predict_binary', methods=['GET', 'POST'])
def predict_binary():
    try:
        if request.method == 'POST':
            # Retrieve the JSON data from the request
            data = request.get_json()

            # Preprocess the input data
            input_data = preprocess_data(data['image_data'])

            # Get the input tensor index
            input_tensor_index = binary_interpreter.get_input_details()[0]['index']

            # Set the input tensor using the global binary interpreter
            binary_interpreter.set_tensor(input_tensor_index, input_data.astype(np.float32))  # Convert to FLOAT32

            # Run inference
            binary_interpreter.invoke()

            # Get the output tensor
            output_tensor_index = binary_interpreter.get_output_details()[0]['index']
            output_data = binary_interpreter.get_tensor(output_tensor_index)

            # Process the output data
            result = process_binary_output(output_data)

            # Return the prediction
            return jsonify(result)

        elif request.method == 'GET':
            # If it's a GET request, return the HTML page
            print("GET request received for /predict_binary")
            return send_from_directory('.', 'binarymodel.html')

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'prediction': 'Error during prediction'})

def process_binary_output(output_data):
    # Assuming output_data is a single probability for the binary classification
    probability_benign = output_data[0][0]

    # Set a threshold for classifying as benign
    threshold = 0.5
    predicted_class_label = 'Benign' if probability_benign >= threshold else 'Non-Benign'

    return {'prediction': predicted_class_label}

# Serve the binary model page
@app.route('/binarymodel')
def binary_model():
    return send_from_directory('.', 'binarymodel.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
