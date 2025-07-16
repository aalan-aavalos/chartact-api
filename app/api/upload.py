from fastapi import APIRouter, UploadFile, File
from app.services.csv_processor import process_csv

router = APIRouter()


@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    df = await process_csv(file)

    preview = df.head().to_dict(orient="records")

    return {
        "filename": file.filename,
        "columns": df.columns.tolist(),
        "preview": preview,
        "shape": df.shape,
    }
