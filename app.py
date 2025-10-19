from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = request.form

    # Preprocess the data to match your model's input requirements
    features = [
        float(data['overallGrade']),
        float(data['livingArea']),
        float(data['bathrooms']),
        float(data['bedrooms']),
        float(data['floors'])
    ]

    # Make prediction using your model
    prediction = model.predict([features])

    # For now, return a dummy prediction
    # Replace this with actual prediction from your model
    # predicted_price = 350000  # This would be your model's output

    return jsonify({'predicted_price': prediction})


if __name__ == '__main__':
    app.run()