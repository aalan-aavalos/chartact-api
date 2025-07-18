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
    train_and_store_model,
)
from app.models import CreateModelResponse

router = APIRouter()


@router.post("/create", response_model=CreateModelResponse)
async def create_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    dimensions: str = Form(...),      # JSON string: ["E", "S", "T", "J"]
    # JSON string: ["Analista", "Centinela", …]
    mbti_categories: str = Form(...),
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

    train_and_store_model(
        labeled_df,  # o labeled_df
        dimensions_list,
        model_name=model_name,
        description=description,
        mbti_categories=mbti_categories_list,
        training_type="clustering"  # o "clustering"
    )

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
    dimensions: str = Form(...),      # Ej: ["E", "S", "T", "J"]
    mbti_categories: str = Form(...),  # Ej: ["Analista", "Centinela"]
    description: str = Form(None)
):
    import json
    from app.services.model_trainer import get_opposite, assign_by_distance

    dimensions_list = json.loads(dimensions)
    mbti_categories_list = json.loads(mbti_categories)

    # 1. Leer CSV original
    df = await process_csv(file)

    # 2. Calcular dimensiones + sus opuestos
    all_needed_dims = set(dimensions_list)
    all_needed_dims.update(get_opposite(d) for d in dimensions_list)
    all_needed_dims = list(all_needed_dims)

    # 3. Prepara DataFrame con esas dimensiones
    prepared_df = prepare_dataframe(df, all_needed_dims)

    # 4. Asignar código MBTI
    prepared_df["mbti_code"] = prepared_df.apply(assign_mbti_code, axis=1)

    # 5. Mapear a categoría MBTI clásica (pueden salir las 4)
    prepared_df["mbti_label"] = prepared_df["mbti_code"].apply(
        map_mbti_to_category)

    # 6. Reasignar etiquetas según cercanía si el usuario no mandó las 4 categorías
    if set(prepared_df["mbti_label"].unique()) - set(mbti_categories_list):
        prepared_df = assign_by_distance(
            prepared_df, list(dimensions_list), mbti_categories_list)

    # 7. Obtener distribución
    label_distribution = prepared_df["mbti_label"].value_counts().to_dict()

    train_and_store_model(
        prepared_df,  # o labeled_df
        dimensions_list,
        model_name=model_name,
        description=description,
        mbti_categories=mbti_categories_list,
        training_type="classical"  # o "clustering"
    )

    # 8. Retornar resultado
    return CreateModelResponse(
        model_name=model_name,
        dimensions=dimensions_list,
        clusters_used=list(label_distribution.keys()),
        label_distribution=label_distribution,
        data=prepared_df.drop(columns=["mbti_code"]).to_dict(orient="records")
    )
