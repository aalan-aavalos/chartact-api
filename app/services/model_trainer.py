def prepare_dataframe(df, dimensions: list[str]):
    # Diccionario con mapeo de preguntas a dimensiones
    dim_map = {
        "E": ["p1", "p2", "p3"],    # Extroversión
        "I": ["p4"],                # Introversión
        "S": ["p5", "p6"],          # Sensación
        "N": ["p7", "p8"],          # Intuición
        "T": ["p9", "p11"],         # Pensamiento
        "F": ["p10", "p12"],        # Sentimiento
        "J": ["p13", "p15"],        # Juicio
        "P": ["p14", "p16"],        # Percepción
    }

    df_out = df[["name", "gender", "age_range"]].copy()

    for dim in dimensions:
        if dim not in dim_map:
            continue
        lado_A = dim_map[dim]
        lado_B = dim_map[get_opposite(dim)]
        df_out[dim] = df[lado_A].sum(axis=1)  # Suma lado A
        df_out[dim] += df[lado_B].sum(axis=1) * 0  # Solo para compatibilidad visual (lado B se usaría si aplicamos diferencia)

    return df_out


def get_opposite(dim: str):
    return {
        "E": "I", "I": "E",
        "S": "N", "N": "S",
        "T": "F", "F": "T",
        "J": "P", "P": "J"
    }[dim]
