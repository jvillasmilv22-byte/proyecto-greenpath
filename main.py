import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

app = FastAPI(title="GreenPath Routing API")
model = YOLO("yolov8n.pt")

CAPACIDAD_MAXIMA = 15
UMBRAL_CONGESTION = 70.0
CLASES_VEHICULOS = {2, 5, 7}  # Autos, buses y camiones


@app.post("/analizar-ruta")
async def analizar_ruta(file: UploadFile = File(...)):
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="El archivo está vacío")

    datos_imagen = np.frombuffer(contents, dtype=np.uint8)

    try:
        imagen = cv2.imdecode(datos_imagen, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise HTTPException(
            status_code=400, detail="El archivo no es una imagen válida"
        ) from error

    if imagen is None:
        raise HTTPException(
            status_code=400, detail="El archivo no es una imagen válida"
        )

    resultado = model(imagen, verbose=False)[0]
    conteo = sum(
        1 for box in resultado.boxes if int(box.cls[0]) in CLASES_VEHICULOS
    )

    porcentaje = (conteo / CAPACIDAD_MAXIMA) * 100
    estado = "EVITAR" if porcentaje >= UMBRAL_CONGESTION else "ACCESIBLE"

    return {
        "vehiculos_detectados": conteo,
        "porcentaje_congestion": round(porcentaje, 2),
        "estado_ruta": estado,
    }