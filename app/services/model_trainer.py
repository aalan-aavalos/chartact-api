import uuid
import pandas as pd
from sklearn.cluster import KMeans
import os
import json
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Diccionario con mapeo de preguntas a dimensiones
DIM_MAP = {
    "E": ["p1", "p2", "p3"],    # Extroversión
    "I": ["p4"],                # Introversión
    "S": ["p5", "p6"],          # Sensación
    "N": ["p7", "p8"],          # Intuición
    "T": ["p9", "p11"],         # Pensamiento
    "F": ["p10", "p12"],        # Sentimiento
    "J": ["p13", "p15"],        # Juicio
    "P": ["p14", "p16"],        # Percepción
}

MBTI_LABELS = {
    ("N", "T"): "Analista",
    ("N", "F"): "Diplomático",
    ("S", "J"): "Centinela",
    ("S", "P"): "Explorador"
}


def get_opposite(dim: str) -> str:
    return {
        "E": "I", "I": "E",
        "S": "N", "N": "S",
        "T": "F", "F": "T",
        "J": "P", "P": "J"
    }[dim]


def prepare_dataframe(df: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    df_out = df[["name", "gender", "age_range"]].copy()

    for dim in dimensions:
        if dim not in DIM_MAP:
            continue
        lado_A = DIM_MAP[dim]
        lado_B = DIM_MAP.get(get_opposite(dim), [])
        df_out[dim] = df[lado_A].sum(axis=1)

    return df_out


def assign_mbti_label(row: dict) -> str:
    n_s = "N" if row.get("N", 0) >= row.get("S", 0) else "S"
    t_f = "T" if row.get("T", 0) >= row.get("F", 0) else "F"
    j_p = "J" if row.get("J", 0) >= row.get("P", 0) else "P"

    if n_s == "N" and t_f == "T":
        return "Analista"
    elif n_s == "N" and t_f == "F":
        return "Diplomático"
    elif n_s == "S" and j_p == "J":
        return "Centinela"
    elif n_s == "S" and j_p == "P":
        return "Explorador"
    else:
        return "Desconocido"


def cluster_and_label(df: pd.DataFrame, dimensions: list[str], mbti_categories: list[str]):
    X = df[dimensions]

    num_clusters = len(mbti_categories)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    cluster_centers = kmeans.cluster_centers_

    cluster_labels = {}
    used_labels = set()

    for idx, center in enumerate(cluster_centers):
        temp_row = {dim: val for dim, val in zip(dimensions, center)}
        temp_label = assign_mbti_label(temp_row)

        # Si ya está usada o no está permitida, usa la siguiente del input
        if temp_label in used_labels or temp_label not in mbti_categories:
            fallback_idx = idx % len(mbti_categories)
            temp_label = mbti_categories[fallback_idx]

        used_labels.add(temp_label)
        cluster_labels[idx] = temp_label

    df["mbti_label"] = df["cluster"].map(cluster_labels)
    df.drop(columns=["cluster"], inplace=True)

    return df, cluster_labels


def assign_mbti_code(row: dict) -> str:
    code = ""
    code += "E" if row.get("E", 0) >= row.get("I", 0) else "I"
    code += "S" if row.get("S", 0) >= row.get("N", 0) else "N"
    code += "T" if row.get("T", 0) >= row.get("F", 0) else "F"
    code += "J" if row.get("J", 0) >= row.get("P", 0) else "P"
    return code


def map_mbti_to_category(code: str) -> str:
    if code[1] == "N" and code[2] == "T":
        return "Analista"
    elif code[1] == "N" and code[2] == "F":
        return "Diplomático"
    elif code[1] == "S" and code[3] == "J":
        return "Centinela"
    elif code[1] == "S" and code[3] == "P":
        return "Explorador"
    else:
        return "Desconocido"


IDEAL_CENTROIDS = {
    "Analista": {"N": 10, "T": 10},
    "Diplomático": {"N": 10, "F": 10},
    "Centinela": {"S": 10, "J": 10},
    "Explorador": {"S": 10, "P": 10},
}


def assign_by_distance(df: pd.DataFrame, dimensions: list[str], categories: list[str]) -> pd.DataFrame:
    def closest_label(row):
        min_dist = float("inf")
        best_label = "Desconocido"

        for label in categories:
            center = IDEAL_CENTROIDS.get(label, {})
            dist = 0
            dims_in_common = [dim for dim in center if dim in row]

            # Si no hay dimensiones comunes, pasar
            if not dims_in_common:
                continue

            for dim in dims_in_common:
                dist += (row.get(dim, 0) - center.get(dim, 0)) ** 2

            if dist < min_dist:
                min_dist = dist
                best_label = label

        return best_label

    df["mbti_label"] = df.apply(closest_label, axis=1)
    return df


def train_and_store_model(
    df, dimensions, model_name, description, mbti_categories, training_type="clustering", file_info=None
):
    from sklearn.preprocessing import LabelEncoder

    X = df[dimensions]
    y = df["mbti_label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    y_pred = model.predict(X)

    # Precisión general
    accuracy = accuracy_score(y_encoded, y_pred)

    # Reporte detallado
    report_dict = classification_report(
        y_encoded, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    # Matriz de confusión
    conf_matrix = confusion_matrix(y_encoded, y_pred).tolist()

    # Generar ID único y timestamp
    model_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Crear nombre de carpeta: modelo_nombre + _ + model_id
    safe_name = model_name.replace(" ", "_")
    dir_name = f"models_store/{safe_name}_{model_id}"
    os.makedirs(dir_name, exist_ok=True)

    # Guardar modelo
    joblib.dump({
        "model": model,
        "label_encoder": le,
        "dimensions": dimensions
    }, f"{dir_name}/model.pkl")

    # Guardar metadatos
    metadata = {
        "model_id": model_id,
        "model_name": model_name,
        "created_at": timestamp,
        "description": description,
        "dimensions": dimensions,
        "categories": mbti_categories,
        "num_samples": len(df),
        "label_distribution": df["mbti_label"].value_counts().to_dict(),
        "age_ranges": sorted(df["age_range"].unique().tolist()),
        "genders": sorted(df["gender"].unique().tolist()),
        "columns_used": df.columns.tolist(),
        "training_type": training_type,
        "training_results": {
            "accuracy": round(accuracy, 4),
            "classification_report": report_dict,
            "confusion_matrix": conf_matrix
        },
        "labels": le.classes_.tolist(),
    }

    if file_info:
        metadata.update(file_info)

    with open(f"{dir_name}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
