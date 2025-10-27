from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Load the pipeline (this will be pipeline_v2.bin in the base Docker image)
with open("pipeline_v2.bin", "rb") as f:
    pipeline = pickle.load(f)

# Create FastAPI app
app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    # Create feature dictionary for DictVectorizer
    features = {
        'lead_source': lead.lead_source,
        'number_of_courses_viewed': lead.number_of_courses_viewed,
        'annual_income': lead.annual_income
    }
    
    # Make prediction
    probability = pipeline.predict_proba([features])[0][1]
    
    return {"probability": probability}

@app.get("/")
def root():
    return {"message": "Lead Scoring ML Model API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)