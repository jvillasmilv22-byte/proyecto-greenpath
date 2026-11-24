from collections import deque
from math import ceil, hypot, isfinite
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

# Parámetros iniciales pendientes de calibración.
BRILLO_MINIMO = 15
BRILLO_MAXIMO = 245
CONTRASTE_MINIMO = 5
TIEMPO_MINIMO_TRAYECTORIA = 0.5
FRACCION_MINIMA_MOVIMIENTO = 0.01

DIRECCIONES = (
    "HACIA ARRIBA",
    "HACIA ABAJO",
    "HACIA IZQUIERDA",
    "HACIA DERECHA",
    "MOVIMIENTO INSUFICIENTE",
    "SIN TRAYECTORIA SUFICIENTE",
)


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
    estado = "EVITAR" if porcentaje >= UMBRAL_CONGESTION else "ACCESIBLE"

    return {
        "vehiculos_detectados": conteo,
        "porcentaje_congestion": round(porcentaje, 2),
        "estado_ruta": estado,
        "motivo": "Clasificación según el conteo y el umbral del prototipo",
    }


def obtener_movimiento(caja, indice, fps, historial, ancho, alto):
    if caja.id is None:
        return None, "SIN TRAYECTORIA SUFICIENTE"

    identificador = int(caja.id.item())
    x1, y1, x2, y2 = caja.xyxy[0].tolist()
    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2

    trayectoria = historial.setdefault(identificador, deque())

    # Reinicia la observación si el vehículo estuvo ausente.
    if trayectoria:
        ausencia = indice - trayectoria[-1][0]
        if ausencia > max(2, round(fps * 0.2)):
            trayectoria.clear()

    trayectoria.append((indice, centro_x, centro_y))

    # Conserva aproximadamente el último segundo.
    while len(trayectoria) > 1 and indice - trayectoria[0][0] > fps:
        trayectoria.popleft()

    primer_indice, primer_x, primer_y = trayectoria[0]
    tiempo_observado = (indice - primer_indice) / fps

    if tiempo_observado < TIEMPO_MINIMO_TRAYECTORIA:
        return identificador, "SIN TRAYECTORIA SUFICIENTE"

    desplazamiento_x = centro_x - primer_x
    desplazamiento_y = centro_y - primer_y
    umbral = hypot(ancho, alto) * FRACCION_MINIMA_MOVIMIENTO

    if hypot(desplazamiento_x, desplazamiento_y) < umbral:
        return identificador, "MOVIMIENTO INSUFICIENTE"

    if abs(desplazamiento_y) >= abs(desplazamiento_x):
        direccion = (
            "HACIA ABAJO"
            if desplazamiento_y > 0
            else "HACIA ARRIBA"
        )
    else:
        direccion = (
            "HACIA DERECHA"
            if desplazamiento_x > 0
            else "HACIA IZQUIERDA"
        )

    return identificador, direccion


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
    if Path(file.filename or "").suffix.lower() != ".mp4":
        raise HTTPException(
            status_code=400,
            detail="Sube un archivo MP4",
        )

    muestras = []
    bytes_recibidos = 0
    fotogramas_procesados = 0

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

            # Modelo independiente para cada solicitud de video.
            modelo_video = YOLO("yolov8n.pt")
            historial = {}

            indices_muestra = {
                round(segundo * fps)
                for segundo in range(MAX_SEGUNDOS_VIDEO)
            }
            limite_fotogramas = ceil(fps * MAX_SEGUNDOS_VIDEO)

            for indice in range(limite_fotogramas):
                leido, fotograma = captura.read()

                if not leido:
                    break

                fotogramas_procesados += 1
                motivo = evaluar_visibilidad(fotograma)

                if motivo is not None:
                    historial.clear()

                    predictor = modelo_video.predictor
                    for tracker in getattr(predictor, "trackers", []):
                        tracker.reset()

                    if indice in indices_muestra:
                        muestras.append({
                            "segundo": round(indice / fps, 2),
                            "vehiculos_detectados": None,
                            "porcentaje_congestion": None,
                            "estado_ruta": "NO EVALUABLE",
                            "motivo": motivo,
                            "movimientos": None,
                            "vehiculos": [],
                        })

                    continue

                # El seguimiento procesa fotogramas consecutivos.
                resultado = modelo_video.track(
                    fotograma,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=sorted(CLASES_VEHICULOS),
                    conf=0.25,
                    verbose=False,
                )[0]

                alto, ancho = fotograma.shape[:2]
                movimientos = dict.fromkeys(DIRECCIONES, 0)
                vehiculos = []

                for caja in resultado.boxes:
                    identificador, direccion = obtener_movimiento(
                        caja,
                        indice,
                        fps,
                        historial,
                        ancho,
                        alto,
                    )

                    movimientos[direccion] += 1
                    vehiculos.append({
                        "id": identificador,
                        "clase": resultado.names[int(caja.cls.item())],
                        "movimiento": direccion,
                    })

                caducados = [
                    identificador
                    for identificador, trayectoria in historial.items()
                    if indice - trayectoria[-1][0] > fps * 2
                ]

                for identificador in caducados:
                    del historial[identificador]

                # Devuelve aproximadamente una muestra por segundo.
                if indice in indices_muestra:
                    conteo = len(vehiculos)
                    porcentaje = conteo / CAPACIDAD_MAXIMA * 100
                    estado = (
                        "EVITAR"
                        if porcentaje >= UMBRAL_CONGESTION
                        else "ACCESIBLE"
                    )

                    muestras.append({
                        "segundo": round(indice / fps, 2),
                        "vehiculos_detectados": conteo,
                        "porcentaje_congestion": round(porcentaje, 2),
                        "estado_ruta": estado,
                        "motivo": (
                            "Clasificación según el conteo "
                            "y el umbral del prototipo"
                        ),
                        "movimientos": movimientos,
                        "vehiculos": vehiculos,
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
        "fotogramas_procesados": fotogramas_procesados,
        "descripcion": (
            "Muestras aproximadamente por segundo y movimiento "
            "respecto de una cámara fija"
        ),
        "muestras": muestras,
    }