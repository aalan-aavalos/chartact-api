# api/pure_mbti.py
import json
from fastapi import APIRouter, UploadFile, File, Form
from app.services.csv_processor import process_csv
from app.services.model_trainer import prepare_dataframe, assign_mbti_code, map_mbti_to_category

router = APIRouter()

@router.post("/pure-mbti")
async def generate_pure_mbti(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    description: str = Form(None)
):
    # 1. Procesar CSV
    df = await process_csv(file)

    # 2. Calcular TODAS las dimensiones
    all_dims = ["E", "I", "S", "N", "T", "F", "J", "P"]
    df_dims = prepare_dataframe(df, all_dims)

    # 3. Asignar código MBTI (ej. ENTP)
    df_dims["mbti_code"] = df_dims.apply(assign_mbti_code, axis=1)

    # 4. Mapear código a categoría macro
    df_dims["mbti_label"] = df_dims["mbti_code"].apply(map_mbti_to_category)

    # 5. Distribución de etiquetas
    distribution = df_dims["mbti_label"].value_counts().to_dict()

    return {
        "model_name": model_name,
        "description": description,
        "total_records": len(df_dims),
        "distribution": distribution,
        "data": df_dims[["name", "gender", "age_range", "mbti_code", "mbti_label"]].to_dict(orient="records")
    }
