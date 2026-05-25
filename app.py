import pickle#
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Diabetes Prediction API")

app.add_middleware (
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("Prototype/Model/diabetes_knn_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
accuracy = model_data["accuracy"]

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API"}

@app.post("/predict")
def predict_diabetes(Data: DiabetesInput):
    # Convert input data to a DataFrame
    df = pd.DataFrame(
        [[
            Data.Pregnancies,
            Data.Glucose,
            Data.BloodPressure,
            Data.SkinThickness,
            Data.Insulin,
            Data.BMI,
            Data.DiabetesPedigreeFunction,
            Data.Age
        ]],
        columns=features
    )

    # Scale the features
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)[0]
    class_label = "Diabetic" if int(prediction[0]) == 1 else "Non-Diabetic"

    return {
        "prediction": int(prediction[0]),
        "class_label": class_label,
        "probability": float(probability[int(prediction[0])])
    }