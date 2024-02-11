from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    
    # Convert the input data to a numpy array
    input_data = np.array(data).reshape(1, -1)
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
