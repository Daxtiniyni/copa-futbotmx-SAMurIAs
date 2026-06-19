# SAMurIAs

Proyecto de visión por computadora para la Copa FutBotMX.

SAMurIAs utiliza SAM 3 y SAM 3.1 para detectar, segmentar y analizar elementos
de partidos de fútbol robótico:

- Robots de ambos equipos.
- Balón.
- Cancha.
- Resultados visuales con máscaras de color.
- Exportación de detecciones y métricas en JSON.

## Estado actual

El prototipo permite:

1. Segmentar imágenes mediante prompts de texto.
2. Identificar robots y balón.
3. Procesar videos de fútbol robótico.
4. Clasificar visualmente robots entre equipo rojo y equipo azul.
5. Generar videos anotados y registros JSON.
6. Explorar resultados desde una plataforma web.
7. Mantener identidades temporales y dibujar trayectorias de robots y balón.
8. Calibrar una cancha fija mediante homografía.
9. Generar mapas de calor sobre un campo canónico y narrativa automática.

## Requisitos

- Python 3.12 o superior.
- PyTorch 2.7 o superior.
- Acceso autorizado a `facebook/sam3` y `facebook/sam3.1` en Hugging Face.
- GPU NVIDIA con CUDA recomendada para video.

El prototipo también puede ejecutarse en CPU sobre macOS, aunque el
procesamiento es considerablemente más lento.

## Instalación

```bash
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate
pip install torch torchvision
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
pip install -e third_party/sam3
pip install numpy==1.26.4 pillow matplotlib opencv-python einops pycocotools psutil
hf auth login
```

SAM 3 se distribuye bajo su propia licencia. Consulta el repositorio oficial
antes de utilizar o redistribuir modelos y pesos.

## Pruebas

Validar la instalación:

```bash
python scripts/check_sam31_install.py --with-weights
```

Segmentar una imagen:

```bash
python scripts/run_sam3_image_prompt.py imagen.jpg \
  --prompt "robot" \
  --out outputs/sam3/robot.png
```

Procesar un video:

```bash
python scripts/segment_robot_soccer_video.py partido.mp4 \
  --seconds 60 \
  --sample-fps 1 \
  --out outputs/sam3/partido_segmentado.mp4 \
  --json-out outputs/sam3/partido_segmentado.json
```

Procesar con homografía:

```bash
python scripts/segment_robot_soccer_video.py partido.mp4 \
  --calibration data/calibrations/partido.json \
  --out outputs/sam3/partido_segmentado.mp4 \
  --json-out outputs/sam3/partido_segmentado.json
```

El archivo de calibración contiene cuatro puntos normalizados entre `0` y `1`,
en orden superior izquierda, superior derecha, inferior derecha e inferior
izquierda:

```json
{
  "points": [[0.1, 0.2], [0.9, 0.2], [0.95, 0.9], [0.05, 0.9]],
  "normalized": true
}
```

## Plataforma web

Iniciar el servidor:

```bash
source .venv-sam3/bin/activate
python app.py
```

Abrir:

```text
http://127.0.0.1:5050
```

La plataforma permite:

- Cargar un video desde el navegador.
- Ejecutar el análisis SAM en segundo plano.
- Reproducir el resultado segmentado.
- Consultar métricas de presencia visual.
- Revisar mapas de calor por equipo.
- Calibrar una cancha de cámara fija seleccionando sus cuatro esquinas.
- Explorar trayectorias persistentes en un mapa táctico de `900 × 600`.
- Generar mapas de calor con coordenadas proyectadas por homografía.
- Leer una narrativa automática del análisis.

Los resultados se almacenan localmente en `outputs/sam3/`. Los videos de entrada,
pesos de modelos y resultados generados están excluidos del repositorio público.

## Referencias del curso FutBotMX

El diseño del pipeline tomó como referencia conceptual los notebooks de formación
compartidos para la competencia, principalmente:

- NB09: segmentación de video, tracking y filtrado previo a SAM.
- NB10: fundamentos de homografía y campo canónico.
- NB11: detección HSV y uso de `BOTTOM_CENTER` para representar robots.
- NB12: acumulación temporal de máscaras y mapas de calor.
- NB13: integración nativa de SAM 3 desde Hugging Face.

La implementación de este repositorio es propia. Los notebooks del curso no se
redistribuyen y permanecen fuera del historial Git.

La homografía implementada supone una cámara fija durante todo el video. Si la
cámara se desplaza, cambia el zoom o gira, se requiere recalibración por tramo o
estimación dinámica de la cancha.

## Colores de visualización

- Equipo rojo: rojo.
- Equipo azul: azul.
- Balón: amarillo.
- Cancha: verde.

## Equipo

SAMurIAs.

## Licencia

El código propio del proyecto se publica bajo la licencia MIT. Las dependencias,
modelos y pesos de terceros mantienen sus licencias originales.
