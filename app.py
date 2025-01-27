from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    try:
        model = pickle.load(file)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading the model:", str(e))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        input_features = [float(x) for x in request.form.values()]
        input_array = np.array([input_features])

        # Make prediction
        prediction = model.predict(input_array)

        # Display result
        result = "Positive" if prediction[0] == 1 else "Negative"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)