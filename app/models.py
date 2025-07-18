from pydantic import BaseModel
from typing import List, Optional, Dict


class CreateModelRequest(BaseModel):
    model_name: str
    description: Optional[str] = None
    dimensions: List[str]               # Ej: ["E", "S", "T", "J"]
    mbti_categories: List[str]          # Ej: ["Analista", "Centinela"]


class ModelMetadata(BaseModel):
    model_name: str
    description: Optional[str]
    created_at: str                     # ISO format o fecha simple
    dimensions: List[str]
    mbti_categories: List[str]
    num_records: int


class CreateModelResponse(BaseModel):
    model_name: str
    dimensions: List[str]
    clusters_used: List[str]            # Ej: ["Analista", "Centinela"]
    label_distribution: Dict[str, int]  # Ej: {"Analista": 20, "Centinela": 30}
    data: List[Dict]                    # Cada fila como dict (nombre, genero, dimensiones, etiqueta)


class CreateModelUseResponse(BaseModel):
    model_id: str  
    model_name: str
    dimensions: List[str]
    clusters_used: List[str]            # Ej: ["Analista", "Centinela"]
    label_distribution: Dict[str, int]  # Ej: {"Analista": 20, "Centinela": 30}
    data: List[Dict]                    # Cada fila como dict (nombre, genero, dimensiones, etiqueta)
