from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    carat = float(data['carat'])
    clarity = data['clarity']
    x = float(data['X'])
    y = float(data['Y'])
    z = float(data['Z'])
    color = data['color']

    # Prepare input for prediction
    input_data = pd.DataFrame([[carat, clarity, x, y, z, color]],
                               columns=['carat', 'clarity', 'x', 'y', 'z', 'color'])

    # Make prediction
    price_prediction = model.predict(input_data)[0]
    price_prediction *= 85
    return jsonify({'predicted_price': round(price_prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
