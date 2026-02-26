from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import RegressionRequest, RegressionResponse, ClassificationRequest, ClassificationResponse
from models import run_regression_logic, run_classification_logic

app = FastAPI(title="ML Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Evaluation API"}

@app.post("/regression", response_model=RegressionResponse)
async def perform_regression(data: RegressionRequest):
    result = run_regression_logic(data.features, data.targets)
    return result

@app.post("/classification", response_model=ClassificationResponse)
async def perform_classification(data: ClassificationRequest):
    result = run_classification_logic(data.features, data.targets)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
