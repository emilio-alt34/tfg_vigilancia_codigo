# TFG Vigilancia Poscomercializacion

Repositorio de codigo del TFG sobre vigilancia poscomercializacion de sistemas de IA aplicados a radiologia musculoesqueletica.

## Objetivo del repositorio

Este repositorio recoge un prototipo academico dividido en dos partes:

- un notebook de entrenamiento e inferencia sobre MURA-v1.1
- una aplicacion en Streamlit para revisar inferencias, registrar valoraciones humanas y vigilar el comportamiento del sistema

## Contenido principal

- `app.py`: interfaz de Streamlit
- `vigilance_backend.py`: logica del sistema de vigilancia
- `01_finetuning_efficientnet_mura.ipynb`: notebook de entrenamiento e inferencia
- `02_prueba_interfaz.ipynb`: notebook de pruebas funcionales de la aplicacion
- `outputs/tablas/`: CSVs generados para ejecutar la app sin repetir todo el entrenamiento
- `outputs/imagenes_analizadas/`: imagenes analizadas que usa la demo visual de la app

## Estructura esperada

```text
.
├── app.py
├── vigilance_backend.py
├── 01_finetuning_efficientnet_mura.ipynb
├── 02_prueba_interfaz.ipynb
├── requirements.txt
├── data/
│   └── README_MURA.txt
└── outputs/
    ├── imagenes_analizadas/
    └── tablas/
```

## Requisitos generales

Se recomienda Python 3.10 o superior.

Las dependencias principales del proyecto se recogen en `requirements.txt`.

## Instalacion

Crear o activar un entorno virtual y despues instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Dataset

El dataset MURA-v1.1 no se incluye en este repositorio.

Si se quiere reproducir el entrenamiento o mostrar las radiografias originales en la app, debe colocarse manualmente en:

`data/MURA-v1.1/`

La estructura esperada se explica en:

`data/README_MURA.txt`

## Que se puede ejecutar sin descargar MURA

La aplicacion puede arrancar directamente con los CSV ya incluidos en `outputs/tablas/`, sin necesidad de repetir el entrenamiento completo.

Si no se descarga MURA:

- la app seguira funcionando
- se podrán revisar metricas, alertas, auditoria e informe
- se podran mostrar las imagenes analizadas ya exportadas en `outputs/imagenes_analizadas/`
- no se podran mostrar las radiografias originales si faltan los archivos de `data/MURA-v1.1/`

## Ejecutar la aplicacion

Lanzar Streamlit desde la raiz del repositorio:

```bash
streamlit run app.py
```

## Reproducir el entrenamiento

Abrir y ejecutar:

`01_finetuning_efficientnet_mura.ipynb`

Este notebook:

- entrena el modelo multitarea basado en EfficientNet-B0
- genera las inferencias del modelo
- exporta las tablas necesarias para la app
- guarda las imagenes analizadas que luego usa la interfaz

Para reproducirlo completamente es necesario disponer del dataset MURA-v1.1 en la ruta esperada.

## Ejecutar las pruebas de la aplicacion

Abrir y ejecutar:

`02_prueba_interfaz.ipynb`

Este notebook prueba la logica principal del sistema de vigilancia mediante escenarios controlados, incluyendo:

- carga inicial del sistema
- flujo de revision clinica
- generacion y cierre de alertas
- auditoria
- informe de vigilancia
- drift simple de distribucion

## Archivos generados relevantes

Dentro de `outputs/tablas/` se incluyen los ficheros principales que usa la aplicacion:

- `registro_inferencias_modelo.csv`
- `feedback_manual.csv`
- `alertas_generadas.csv`
- `alertas_estado.csv`
- `audit_log.csv`

## Notas de uso

- La interfaz esta pensada como prototipo academico, no como producto clinico real.
- La logica principal vive en `vigilance_backend.py`, `app.py` se ocupa sobre todo de la visualizacion.
- Si se regeneran las inferencias desde el notebook, las tablas de `outputs/tablas/` se actualizaran.
- Si se quiere compartir una demo ligera, puede usarse la app con los CSV ya incluidos, sin necesidad de relanzar el notebook.

## Posibles incidencias

- Si Streamlit no encuentra una imagen original, normalmente faltara `data/MURA-v1.1/`.
- Si la app no arranca, comprobar que las dependencias estan instaladas y que `outputs/tablas/` contiene los CSV esperados.
- Si se quiere volver a generar todo desde cero, primero debe ejecutarse el notebook de entrenamiento.
