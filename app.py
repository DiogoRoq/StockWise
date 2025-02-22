from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from demand_prediction import train_model, predict_demand
from io import StringIO

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup if exists
MODEL_PATH = "demand_prediction_model.pkl"

# Global variable to store RMSE
TRAINING_RESULTS = {"rmse": None, "status": "idle"}

def train_and_save(data):
    """
    Train the model and save it to disk.
    """
    global TRAINING_RESULTS
    try:
        model, feature_scaler, target_scaler, feature_names, rmse = train_model(data)
        joblib.dump((model, feature_scaler, target_scaler, feature_names), MODEL_PATH)
        TRAINING_RESULTS = {"rmse": rmse, "status": "success"}  # Update global variable
    except Exception as e:
        TRAINING_RESULTS = {"rmse": None, "status": f"error: {str(e)}"}  # Update global variable with error  # Update global variable with error

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        contents = await file.read()
        data = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Reset the global variable
        global TRAINING_RESULTS
        TRAINING_RESULTS = {"rmse": None, "status": "training"}

        # Train and save model in background
        background_tasks.add_task(train_and_save, data)  # Pass `data` as an argument
        return {"message": "Model training started in the background!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/training-status/")
async def get_training_status():
    global TRAINING_RESULTS
    if TRAINING_RESULTS["status"] == "success":
        return {"status": "success", "rmse": TRAINING_RESULTS["rmse"]}
    elif TRAINING_RESULTS["status"] == "training":
        return {"status": "training", "message": "Model training is in progress."}
    elif TRAINING_RESULTS["status"].startswith("error"):
        return {"status": "error", "message": TRAINING_RESULTS["status"]}
    else:
        return {"status": "idle", "message": "No training has been performed yet."}

# Add prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Check if the model exists
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model not trained yet. Please upload training data first.")

        # Load the trained model, scalers, and feature names
        model, feature_scaler, target_scaler, feature_columns = joblib.load(MODEL_PATH)
        print("Feature columns from saved model:", feature_columns)
        # Read the uploaded file
        contents = await file.read()
        data = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Make predictions using the predict_demand function
        predictions = predict_demand(model, data, feature_scaler, target_scaler, feature_columns)

        # Return predictions as JSON
        return {"predictions": predictions.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))