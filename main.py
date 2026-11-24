from math import ceil, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

app = FastAPI(title="GreenPath Routing API")
model = YOLO("yolov8n.pt")
bloqueo_modelo = Lock()

CAPACIDAD_MAXIMA = 15
UMBRAL_CONGESTION = 70.0
CLASES_VEHICULOS = {2, 5, 7}  # Autos, buses y camiones

MAX_BYTES_VIDEO = 50 * 1024 * 1024
MAX_SEGUNDOS_VIDEO = 30
TAMANO_BLOQUE = 1024 * 1024

# Parámetros iniciales de visibilidad, pendientes de calibración.
BRILLO_MINIMO = 15
BRILLO_MAXIMO = 245
CONTRASTE_MINIMO = 5


def evaluar_visibilidad(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    brillo_promedio = float(np.mean(gris))
    contraste = float(np.std(gris))

    if brillo_promedio < BRILLO_MINIMO:
        return "La imagen presenta iluminación insuficiente"

    if brillo_promedio > BRILLO_MAXIMO:
        return "La imagen presenta sobreexposición"

    if contraste < CONTRASTE_MINIMO:
        return "La imagen presenta contraste insuficiente"

    return None


def analizar_fotograma(imagen):
    motivo = evaluar_visibilidad(imagen)

    if motivo is not None:
        return {
            "vehiculos_detectados": None,
            "porcentaje_congestion": None,
            "estado_ruta": "NO EVALUABLE",
            "motivo": motivo,
        }

    with bloqueo_modelo:
        resultado = model(imagen, verbose=False)[0]

    conteo = sum(
        1
        for caja in resultado.boxes
        if int(caja.cls[0]) in CLASES_VEHICULOS
    )

    porcentaje = conteo / CAPACIDAD_MAXIMA * 100
    estado = (
        "EVITAR"
        if porcentaje >= UMBRAL_CONGESTION
        else "ACCESIBLE"
    )

    return {
        "vehiculos_detectados": conteo,
        "porcentaje_congestion": round(porcentaje, 2),
        "estado_ruta": estado,
        "motivo": "Clasificación según el conteo y el umbral del prototipo",
    }


@app.post("/analizar-ruta")
def analizar_ruta(file: UploadFile = File(...)):
    contenido = file.file.read()

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail="El archivo está vacío",
        )

    datos_imagen = np.frombuffer(contenido, dtype=np.uint8)

    try:
        imagen = cv2.imdecode(datos_imagen, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise HTTPException(
            status_code=400,
            detail="El archivo no es una imagen válida",
        ) from error

    if imagen is None:
        raise HTTPException(
            status_code=400,
            detail="El archivo no es una imagen válida",
        )

    return analizar_fotograma(imagen)


@app.post("/analizar-video")
def analizar_video(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension != ".mp4":
        raise HTTPException(
            status_code=400,
            detail="Sube un archivo MP4",
        )

    muestras = []
    bytes_recibidos = 0

    with TemporaryDirectory() as carpeta:
        ruta_video = Path(carpeta) / "entrada.mp4"

        with ruta_video.open("wb") as destino:
            while bloque := file.file.read(TAMANO_BLOQUE):
                bytes_recibidos += len(bloque)

                if bytes_recibidos > MAX_BYTES_VIDEO:
                    raise HTTPException(
                        status_code=413,
                        detail="El video supera los 50 MiB",
                    )

                destino.write(bloque)

        if bytes_recibidos == 0:
            raise HTTPException(
                status_code=400,
                detail="El archivo está vacío",
            )

        captura = cv2.VideoCapture(str(ruta_video))

        try:
            if not captura.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo abrir el video MP4",
                )

            fps = captura.get(cv2.CAP_PROP_FPS)

            if not isfinite(fps) or fps <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo determinar el FPS del video",
                )

            # Selecciona aproximadamente un fotograma por segundo.
            indices_muestra = {
                round(segundo * fps)
                for segundo in range(MAX_SEGUNDOS_VIDEO)
            }

            limite_fotogramas = ceil(fps * MAX_SEGUNDOS_VIDEO)

            for indice in range(limite_fotogramas):
                if indice not in indices_muestra:
                    if not captura.grab():
                        break
                    continue

                leido, fotograma = captura.read()

                if not leido:
                    break

                muestras.append({
                    "segundo": round(indice / fps, 2),
                    **analizar_fotograma(fotograma),
                })

        finally:
            captura.release()

    if not muestras:
        raise HTTPException(
            status_code=400,
            detail="El video no tiene fotogramas legibles",
        )

    return {
        "fotogramas_analizados": len(muestras),
        "muestras": muestras,
    }