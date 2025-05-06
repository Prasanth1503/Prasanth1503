from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(_name_)
model = joblib.load('model/house_price_model.pkl')

@app.route('/')
def home():
    return "House Price Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'predicted_price': prediction[0]})

if _name_ == '_main_':
    app.run(debug=True)
