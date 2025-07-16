import json
from fastapi import APIRouter, UploadFile, File, Form
from app.services.csv_processor import process_csv
from app.services.model_trainer import prepare_dataframe

router = APIRouter()


@router.post("/create")
async def create_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    # Recibe JSON string (Ej: '["E", "S", "T", "J"]')
    dimensions: str = Form(...),
    # Recibe JSON string (Ej: '["Analista", "Centinela"]')
    mbti_categories: str = Form(...),
    description: str = Form(None)
):
    print(model_name, dimensions, mbti_categories, description)
    # 1. Parsear dimensiones y categorías
    dimensions_list = json.loads(dimensions)
    mbti_categories_list = json.loads(mbti_categories)

    # 2. Procesar CSV
    df = await process_csv(file)

    # 3. Preparar DataFrame para clustering
    prepared = prepare_dataframe(df, dimensions_list)

    return {
        "preview": prepared.head().to_dict(orient="records"),
        "shape": prepared.shape,
        "dimensions": dimensions_list,
        "mbti_categories": mbti_categories_list,
        "message": "Datos preparados correctamente. Listo para clustering."
    }
