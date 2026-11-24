from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def codificar_imagen(imagen):
    correcto, contenido = cv2.imencode(".png", imagen)
    assert correcto
    return contenido.tobytes()


def test_ruta_raiz():
    respuesta = client.get("/")
    assert respuesta.status_code == 404


def test_solicitud_sin_archivo():
    respuesta = client.post("/analizar-ruta")
    assert respuesta.status_code == 422


def test_archivo_vacio():
    respuesta = client.post(
        "/analizar-ruta",
        files={"file": ("vacio.png", b"", "image/png")},
    )

    assert respuesta.status_code == 400
    assert respuesta.json()["detail"] == "El archivo está vacío"


def test_archivo_de_texto():
    respuesta = client.post(
        "/analizar-ruta",
        files={
            "file": (
                "prueba.txt",
                b"Este archivo es de texto",
                "text/plain",
            )
        },
    )

    assert respuesta.status_code == 400
    assert respuesta.json()["detail"] == "El archivo no es una imagen válida"


@pytest.mark.parametrize(
    "intensidad,motivo",
    [
        (0, "La imagen presenta iluminación insuficiente"),
        (255, "La imagen presenta sobreexposición"),
        (120, "La imagen presenta contraste insuficiente"),
    ],
)
def test_imagen_sin_visibilidad(monkeypatch, intensidad, motivo):
    def modelo_no_debe_ejecutarse(*args, **kwargs):
        pytest.fail("Se ejecutó YOLO sobre una imagen no evaluable")

    monkeypatch.setattr(main, "model", modelo_no_debe_ejecutarse)

    imagen = np.full(
        (100, 100, 3),
        intensidad,
        dtype=np.uint8,
    )

    respuesta = client.post(
        "/analizar-ruta",
        files={
            "file": (
                "imagen.png",
                codificar_imagen(imagen),
                "image/png",
            )
        },
    )

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "vehiculos_detectados": None,
        "porcentaje_congestion": None,
        "estado_ruta": "NO EVALUABLE",
        "motivo": motivo,
    }


@pytest.mark.parametrize(
    "cantidad,estado,porcentaje",
    [
        (0, "ACCESIBLE", 0.0),
        (10, "ACCESIBLE", 66.67),
        (11, "EVITAR", 73.33),
    ],
)
def test_regla_del_endpoint(monkeypatch, cantidad, estado, porcentaje):
    # Simula detecciones de autos para comprobar la regla del endpoint.
    cajas = [
        SimpleNamespace(cls=[2])
        for _ in range(cantidad)
    ]

    # Una persona debe quedar fuera del conteo de vehículos.
    cajas.append(SimpleNamespace(cls=[0]))

    def modelo_simulado(imagen, verbose=False):
        return [SimpleNamespace(boxes=cajas)]

    monkeypatch.setattr(main, "model", modelo_simulado)

    # Imagen con brillo y contraste suficientes para pasar el filtro.
    imagen = np.zeros((100, 100, 3), dtype=np.uint8)
    imagen[:, 50:] = 200

    respuesta = client.post(
        "/analizar-ruta",
        files={
            "file": (
                "imagen.png",
                codificar_imagen(imagen),
                "image/png",
            )
        },
    )

    assert respuesta.status_code == 200

    resultado = respuesta.json()
    assert resultado["vehiculos_detectados"] == cantidad
    assert resultado["porcentaje_congestion"] == porcentaje
    assert resultado["estado_ruta"] == estado