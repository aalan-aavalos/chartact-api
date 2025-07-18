# api/models.py
from fastapi import HTTPException
import json
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
from app.services.csv_processor import process_csv
from app.services.model_trainer import (
    prepare_dataframe,
    assign_by_distance,
    assign_mbti_code,
    map_mbti_to_category,
    get_opposite,
    train_and_store_model,
)
from app.models import CreateModelResponse, CreateModelUseResponse
import joblib


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


@router.get("/list")
async def list_models():
    base_path = "models_store"
    if not os.path.exists(base_path):
        return JSONResponse(content={"models": []})

    model_dirs = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    models = []
    for dir_name in sorted(model_dirs, reverse=True):
        metadata_path = os.path.join(base_path, dir_name, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
                models.append(metadata)

    return {"models": models}


@router.delete("/delete/{model_id}")
async def delete_model(model_id: str):
    dir_path = find_model_dir_by_id(model_id)

    # dir_path = f"models_store/{model_id}"

    if not os.path.exists(dir_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    import shutil
    shutil.rmtree(dir_path)
    return {"message": f"Modelo {model_id} eliminado correctamente"}


@router.post("/apply/{model_id}", response_model=CreateModelUseResponse)
async def apply_model(model_id: str, file: UploadFile = File(...)):
    model_dir = find_model_dir_by_id(model_id)
    model_path = f"{model_dir}/model.pkl"
    metadata_path = f"{model_dir}/metadata.json"

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    # Cargar modelo y metadata
    model_data = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    from app.services.csv_processor import process_csv
    from app.services.model_trainer import prepare_dataframe

    df = await process_csv(file)

    dimensions = metadata["dimensions"]
    df_prepared = prepare_dataframe(df, dimensions)

    # Predecir
    model = model_data["model"]
    le = model_data["label_encoder"]
    X = df_prepared[dimensions]

    y_pred = model.predict(X)
    labels = le.inverse_transform(y_pred)
    df_prepared["mbti_label"] = labels

    label_distribution = df_prepared["mbti_label"].value_counts().to_dict()
    clusters_used = sorted(df_prepared["mbti_label"].unique().tolist())

    return CreateModelUseResponse(
        model_id=model_id,
        model_name=metadata["model_name"],
        dimensions=dimensions,
        clusters_used=clusters_used,
        label_distribution=label_distribution,
        data=df_prepared.to_dict(orient="records")
    )


def find_model_dir_by_id(model_id: str) -> str:
    base_path = "models_store"
    for dirname in os.listdir(base_path):
        if model_id in dirname:
            return os.path.join(base_path, dirname)
    raise FileNotFoundError(f"Modelo con id {model_id} no encontrado")
