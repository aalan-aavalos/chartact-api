from fastapi import FastAPI, Response
from app.api import upload, models, pure_mbti

app = FastAPI(title="Charact API")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/favicon.ico")
async def favicon():
    return Response(content="", media_type="image/x-icon")


app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(pure_mbti.router, prefix="/api/mbti", tags=["MBTI"])
app.include_router(models.router, prefix="/api/models", tags=["Model"])
