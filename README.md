# GreenPath

Prototipo que estima la congestión vehicular visible en una imagen de tráfico
para apoyar la evaluación de un tramo en el contexto del enrutamiento de ambulancias.

## Cómo ejecutar el prototipo

Desde la carpeta del proyecto, con Docker Desktop iniciado:

```powershell
docker build -t greenpath-api .
docker run --rm -p 8000:8000 greenpath-api
```

Abre [Swagger UI](http://localhost:8000/docs). En `POST /analizar-ruta`,
selecciona **Try it out**, carga una imagen en el campo `file` y pulsa **Execute**.

## Cómo funciona

FastAPI recibe la imagen, OpenCV la decodifica y YOLOv8n detecta autos, buses
y camiones. La API analiza la imagen completa.

La carga se calcula como `vehículos detectados / 15 × 100`. Si alcanza o supera
el 70%, la API devuelve `EVITAR`. En caso contrario, devuelve `ACCESIBLE`.
Como el cálculo usa 15 vehículos como referencia, la carga puede superar el 100%.

## Validación

- HTTP 400 al enviar un archivo de texto.
- HTTP 422 al enviar una solicitud sin el campo `file`.

## Limitaciones

La visibilidad, la oclusión y la distancia pueden afectar el conteo. Si una
imagen muestra ambos sentidos de tránsito, la API puede contar vehículos de ambos.
GreenPath es un prototipo académico de apoyo a la evaluación de un tramo.