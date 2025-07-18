# api/models.py

import json
from fastapi import APIRouter, UploadFile, File, Form
from app.services.csv_processor import process_csv
from app.services.model_trainer import (
    prepare_dataframe,
    assign_by_distance,
    assign_mbti_code,
    map_mbti_to_category,
    get_opposite,
    # train_and_store_model,
)
from app.models import CreateModelResponse

router = APIRouter()


@router.post("/create", response_model=CreateModelResponse)
async def create_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    dimensions: str = Form(...),      # JSON string: ["E", "S", "T", "J"]
    mbti_categories: str = Form(...), # JSON string: ["Analista", "Centinela", …]
    description: str = Form(None)
):
    dimensions_list = json.loads(dimensions)
    mbti_categories_list = json.loads(mbti_categories)

    # 1. Leer y limpiar CSV
    df = await process_csv(file)
    prepared_df = prepare_dataframe(df, dimensions_list)

    # 2. Clasificación con KMeans
    from app.services.model_trainer import cluster_and_label
    labeled_df, cluster_map = cluster_and_label(
        prepared_df.copy(),
        dimensions_list,
        mbti_categories_list
    )

    label_distribution = labeled_df["mbti_label"].value_counts().to_dict()

    return CreateModelResponse(
        model_name=model_name,
        dimensions=dimensions_list,
        clusters_used=list(set(cluster_map.values())),
        label_distribution=label_distribution,
        data=labeled_df.to_dict(orient="records")
    )


@router.post("/create_classical", response_model=CreateModelResponse)
async def create_classical_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    dimensions: str = Form(...),      # JSON string: ["E", "S", "T", "J"]
    mbti_categories: str = Form(...), # JSON string: ["Analista", "Centinela", …]
    description: str = Form(None)
):
    dimensions_list = json.loads(dimensions)
    mbti_categories_list = json.loads(mbti_categories)

    # 1. Procesar CSV
    df = await process_csv(file)

    # 2. Calcular SIEMPRE las 8 dimensiones (como pure_mbti)
    all_needed_dims = set(dimensions_list)
    for dim in dimensions_list:
        all_needed_dims.add(get_opposite(dim))
    all_needed_dims = list(all_needed_dims)
    
    prepared_df = prepare_dataframe(df, all_needed_dims)


    # 3. Asignar código MBTI y categoría
    prepared_df["mbti_code"] = prepared_df.apply(assign_mbti_code, axis=1)
    prepared_df["mbti_label"] = prepared_df["mbti_code"].apply(map_mbti_to_category)

    # 4. Reasignar si hay menos de 4 categorías
    if len(mbti_categories_list) < 4:
        from app.services.model_trainer import assign_by_distance
        prepared_df = assign_by_distance(prepared_df, all_needed_dims, mbti_categories_list)

    # 5. Filtrar por categorías solicitadas
    prepared_df = prepared_df[prepared_df["mbti_label"].isin(mbti_categories_list)]

    # 6. Distribución
    label_distribution = prepared_df["mbti_label"].value_counts().to_dict()

    return CreateModelResponse(
        model_name=model_name,
        dimensions=dimensions_list,
        clusters_used=list(label_distribution.keys()),
        label_distribution=label_distribution,
        data=prepared_df.drop(columns=["mbti_code"]).to_dict(orient="records")
    )