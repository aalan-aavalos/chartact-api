import pandas as pd
from fastapi import UploadFile
from io import StringIO
import unidecode


async def process_csv(file: UploadFile):
    if not file.filename.endswith('.csv'):
        return {"error": "Solo se permiten archivos CSV"}

    contents = await file.read()
    content_str = contents.decode("utf-8")
    df = pd.read_csv(StringIO(content_str))

    # Renombrar columnas
    df.columns.values[0:4] = ['date', 'name', 'age_range', 'gender']
    df.columns.values[4:20] = [f"p{i}" for i in range(1, 17)]

    # Normalizar y filtrar género (mantener solo 'Femenino' y 'Masculino')
    df['gender'] = df['gender'].str.strip().str.lower().map({
        'femenino': 'F',
        'masculino': 'M'
    })
    df = df[df['gender'].notna()]  # Eliminar registros con género no válido

    # Limpiar nombre
    df['name'] = df['name'].str.strip().str.title()       # Capitalizar
    df['name'] = df['name'].str.split().str[0]            # Tomar primer nombre
    df['name'] = df['name'].apply(unidecode.unidecode)    # Quitar acentos
    # Eliminar nombres cortos
    df = df[df['name'].str.len() > 3]

    # Limpiar rango de edad
    df['age_range'] = df['age_range'].str.replace(
        'años', '', regex=False).str.strip()

    # Eliminar filas con valores faltantes (ya depurados)
    df.dropna(inplace=True)

    return df
