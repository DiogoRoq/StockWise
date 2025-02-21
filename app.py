from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import joblib
import os
from demand_prediction import train_model, predict_demand

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        data = pd.read_csv(file)
        model, feature_scaler, target_scaler = train_model(data)
        joblib.dump((model, feature_scaler, target_scaler), 'demand_prediction_model.pkl')
        flash('Model trained and saved successfully!')
        return redirect(url_for('index'))
    flash('Failed to upload file')
    return redirect(request.url)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        try:
            model, feature_scaler, target_scaler = joblib.load('demand_prediction_model.pkl')
            stock_data = pd.read_csv(file)
            predictions = predict_demand(model, stock_data, feature_scaler, target_scaler)
            return render_template('predict.html', predictions=predictions.to_html(classes='sortable'))
        except FileNotFoundError:
            flash('The demand_prediction_model.pkl file is missing. Please train the model first.')
            return redirect(url_for('index'))
    flash('Failed to upload file')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)