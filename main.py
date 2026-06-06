from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class HouseData(BaseModel):
    size: float
    bedrooms: int
    bathrooms: float

@app.post("/predict")
def predict(data: HouseData):
    input_data = [[data.size, data.bedrooms, data.bathrooms]]
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0])}

@app.get("/")
def root():
    return {"message": "Day 2 ML API is Live"}
