from pydantic import BaseModel
from typing import List, Dict, Optional

class RegressionRequest(BaseModel):
    features: List[List[float]]
    targets: List[float]

class RegressionResponse(BaseModel):
    mse: float
    mae: float
    r2: float
    predictions: List[float]

class ClassificationRequest(BaseModel):
    features: List[List[float]]
    targets: List[int]

class ClassificationResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    predictions: List[int]
