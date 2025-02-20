from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
from demand_prediction import train_model, predict_demand

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        data = pd.read_csv(file)
        model = train_model(data)
        joblib.dump(model, 'demand_prediction_model.pkl')
        return redirect(url_for('predict'))
    return redirect(request.url)

@app.route('/predict')
def predict():
    model = joblib.load('demand_prediction_model.pkl')
    stock_data = pd.read_csv('stock_database.csv')
    predictions = predict_demand(model, stock_data)
    return predictions.to_html()

if __name__ == '__main__':
    app.run(debug=True)