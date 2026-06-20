# SAMurIAs - FutBotMX SAM3 Match Analyzer

Plataforma de análisis automático de partidos de fútbol robótico desarrollada
por el equipo **SAMurIAs** para la **Copa FutBotMX 2026**.

La solución integra detección y segmentación, seguimiento multiobjeto,
analítica deportiva, generación de narraciones y síntesis de voz. Su componente
central es Segment Anything Model 3 (SAM3), utilizado para refinar las
segmentaciones generadas a partir de las detecciones de un modelo YOLO11-Seg
especializado.

## Equipo

- Guillermo Dávalos Gutiérrez
- Andrea Guadalupe Lara González
- Marco Antonio Pérez Doimeadios
- Arturo Morales Téllez

## Objetivo

Transformar un video de fútbol robótico en una experiencia completa de análisis
deportivo asistido por inteligencia artificial:

- Detectar robots, balón y porterías.
- Seguir objetos a lo largo del partido.
- Identificar posesiones, pases, intercepciones, tiros, goles y colisiones.
- Calcular métricas de movimiento, posesión y distribución territorial.
- Generar mapas de calor y trayectorias.
- Producir narraciones automáticas y convertirlas a voz.
- Presentar resultados en un dashboard interactivo.

## Arquitectura

```text
Video del partido
        |
        v
Construcción del dataset
        |
        v
Fine-tuning YOLO11-Seg
        |
        v
Detección de objetos
        |
        v
Refinamiento de máscaras con SAM3
        |
        v
Tracking multiobjeto con ByteTrack
        |
        v
Métricas y detección de eventos
        |
        v
Narración y síntesis de voz
        |
        v
Dashboard interactivo
```

## Metodología

### 1. Construcción del dataset

Se extrajeron cuadros representativos de los videos y se anotaron máscaras para
tres clases:

- `ball`
- `goal`
- `robot`

El dataset incluido contiene 56 imágenes anotadas y 264 instancias segmentadas.
Las anotaciones originales están en `sam_masks/` y su conversión a YOLO
Segmentation se encuentra en `data/yolo_seg/`.

### 2. Fine-tuning

El detector YOLO11-Seg fue ajustado con datos específicos de fútbol robótico.
El checkpoint seleccionado se incluye en:

```text
models/futbotmx_yolo11_seg_best.pt
```

Los principales resultados de entrenamiento se encuentran en
`artifacts/training/`.

### 3. Integración de SAM3

El pipeline combina la velocidad de YOLO con el refinamiento espacial de SAM3:

```text
YOLO fine-tuned -> bounding boxes -> SAM3 -> máscaras refinadas
```

El checkpoint de SAM3 no se distribuye en este repositorio. Debe descargarse
desde la fuente oficial correspondiente y colocarse en:

```text
models/sam3.pt
```

### 4. Tracking y analítica

ByteTrack mantiene identidades temporales para robots y balón. Cada registro
incluye ID, clase, confianza, posición, tiempo y trayectoria. Esos registros
alimentan:

- Distancia recorrida.
- Velocidad promedio y máxima.
- Posesión por equipo y robot.
- Actividad por zonas.
- Mapas de calor.
- Detección de eventos.

### 5. Narración y plataforma

Los eventos se convierten en comentarios sincronizados. ElevenLabs puede
generar voz cuando se proporciona una clave mediante variables de entorno. El
dashboard principal está construido con Streamlit.

## Estructura

```text
.
├── artifacts/
│   └── training/              # Métricas y gráficas del entrenamiento
├── data/
│   └── yolo_seg/              # Dataset YOLO train/val
├── models/
│   └── futbotmx_yolo11_seg_best.pt
├── sam_masks/                 # Máscaras y manifiesto de anotación
├── src/                       # Pipeline y aplicaciones
├── .env.example
├── requirements.txt
└── README.md
```

Los videos originales, audios, checkpoints de terceros y los más de 2 GB de
resultados generados no se incluyen en Git. Las ejecuciones crean la carpeta
`outputs/` localmente.

## Requisitos

- Python 3.11 o 3.12 recomendado.
- GPU NVIDIA con CUDA para procesamiento completo de video.
- 16 GB de VRAM recomendados.
- 32 GB de RAM recomendados.
- Acceso autorizado a los pesos de SAM3.

El equipo de desarrollo utilizó una NVIDIA GeForce RTX 5070 Ti con CUDA 12.8.

## Instalación

```bash
git clone https://github.com/Daxtiniyni/copa-futbotmx-SAMurIAs.git
cd copa-futbotmx-SAMurIAs

python -m venv .venv
```

Activación en Windows:

```powershell
.venv\Scripts\activate
```

Activación en macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Para la narración con ElevenLabs:

```bash
cp .env.example .env
```

Después agrega `ELEVENLABS_API_KEY` y, opcionalmente,
`ELEVENLABS_VOICE_ID` en `.env`.

## Preparación

1. Coloca los videos como `data/videos/V0.MOV` y `data/videos/V1.MOV`.
2. Coloca el checkpoint oficial de SAM3 como `models/sam3.pt`.
3. Verifica que `models/futbotmx_yolo11_seg_best.pt` esté disponible.
4. Ajusta los parámetros de entrada al comienzo de cada script si los nombres
   de tus archivos son diferentes.

## Reproducción paso a paso

Extraer y seleccionar cuadros:

```bash
python src/01_extract_frames.py
python src/02_select_frames_to_label.py
```

Convertir las máscaras al formato YOLO:

```bash
python src/03_convert_manifest_to_yolo_seg.py
```

Entrenar el modelo:

```bash
python src/src04_train_yolo_seg.py
```

Probar SAM3:

```bash
python src/09_test_sam3_load.py
python src/10_test_sam3_bbox.py
```

Ejecutar detección, refinamiento SAM3 y tracking:

```bash
python src/11_yolo_sam3_track_video.py
```

Generar analítica y visualizaciones:

```bash
python src/07_analyze_tracks.py
python src/08_make_visualizations.py
python src/14_detect_events.py
```

Generar narración y voz:

```bash
python src/15_generate_commentary.py
python src/16_elevenlabs_tts.py
python src/17_make_final_video.py
```

Ejecutar el dashboard:

```bash
streamlit run src/futbotmx_streamlit_dashboard_final.py
```

## Módulos principales

- `03_convert_manifest_to_yolo_seg.py`: conversión reproducible del dataset.
- `src04_train_yolo_seg.py`: fine-tuning de YOLO11-Seg.
- `11_yolo_sam3_track_video.py`: YOLO, SAM3 y ByteTrack.
- `07_analyze_tracks.py`: métricas derivadas de trayectorias.
- `08_make_visualizations.py`: mapas de calor y trayectorias.
- `14_detect_events.py`: inferencia de eventos deportivos.
- `15_generate_commentary.py`: narración sincronizada.
- `16_elevenlabs_tts.py`: síntesis de voz.
- `17_make_final_video.py`: composición audiovisual final.
- `futbotmx_streamlit_dashboard_final.py`: plataforma interactiva.

## Artefactos generados

Durante la ejecución se crean, entre otros:

```text
outputs/
├── analysis/
├── events/
├── final/
├── heatmaps/
├── narration/
├── runs/
├── sam3_pipeline_v0/
├── tracking/
└── visualizations/
```

Estos directorios pueden contener videos y CSV de gran tamaño, por lo que están
excluidos del repositorio.

## Principales contribuciones

- Integración de SAM3 dentro de un pipeline completo de análisis deportivo.
- Dataset especializado de fútbol robótico con máscaras por instancia.
- Fine-tuning de YOLO11-Seg para robots, balón y porterías.
- Tracking multiobjeto con ByteTrack.
- Métricas tácticas y temporales.
- Detección automática de eventos deportivos.
- Narración generativa y síntesis de voz.
- Dashboard interactivo para explorar el partido.

## Reel de Instagram

Enlace público solicitado por la convocatoria:

`https://www.instagram.com/reel/DZy-nizNaYF/?igsh=MWsyNXNmZGVpdGZtZw==`

## Licencias

El código propio se distribuye bajo licencia MIT. Los modelos, datasets,
servicios, audios y dependencias de terceros conservan sus licencias y
condiciones de uso originales.
