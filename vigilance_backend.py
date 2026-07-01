"""
Motor de vigilancia poscomercializacion del prototipo.

Este modulo contiene carga de datos, fusion de feedback humano, metricas, calibracion, 
alertas emergentes, auditoria y reporte. La app de Streamlit solo llama a estas funciones 
para visualizar el sistema.
"""

from datetime import datetime, timezone
from pathlib import Path  # Trabajar con rutas
from uuid import uuid4   # Para generar identificadores únicos para el feedback y las alertas

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Rutas
DIR_REPO = Path(__file__).resolve().parent
DIR_DATA = DIR_REPO / "data"
DIR_OUTPUTS = DIR_REPO / "outputs"
DIR_TABLAS = DIR_OUTPUTS / "tablas"
DIR_MURA = DIR_DATA / "MURA-v1.1"

ARCHIVO_REGISTRO_INFERENCIAS = DIR_TABLAS / "registro_inferencias_modelo.csv" # Archivo generado por el notebook
ARCHIVO_REGISTRO_INFERENCIAS_ANTIGUO = DIR_TABLAS / "tabla_base.csv"          # Nombre anterior, mantenido por compatibilidad
ARCHIVO_FEEDBACK = DIR_TABLAS / "feedback_manual.csv"                         # Archivo donde se guardan las valoraciones humanas
ARCHIVO_ALERTAS = DIR_TABLAS / "alertas_generadas.csv"                        # Archivo donde se guardan las alertas generadas
ARCHIVO_ESTADO_ALERTAS = DIR_TABLAS / "alertas_estado.csv"                    # Archivo donde se guardan los estados de cada alerta
ARCHIVO_AUDITORIA = DIR_TABLAS / "audit_log.csv"                              # Archivo histórico de auditoría

MIN_CASOS_ALERTA = 5           # El sistema no genera alertas hasta que haya al menos 5 casos revisados
UMBRAL_ALTA_CONFIANZA = 0.80   # Si el modelo se equivoca con una confianza >= 80%, se considera un error de alta confianza
VERSION_MODELO = "EfficientNet-B0 fine-tuned MURA v1.2"
RANGOS_PROBABILIDAD = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ETIQUETAS_RANGOS_PROBABILIDAD = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

# Definimos las etiquetas que se usan en la app para mostrar los estados, decisiones, actores, eventos, etc.
ETIQUETAS_DECISION = {
    "agree": "Acuerdo con la prediccion",
    "disagree": "Discrepancia",
    "not_reviewed": "Pendiente",
}
ETIQUETAS_REVISION = {
    "none": "Sin incidencia",
    "doubtful": "Caso dudoso",
    "poor_image_quality": "Mala calidad de imagen",
    "out_of_scope_finding": "Hallazgo fuera de alcance",
}
ETIQUETAS_ESTADO = {
    "open": "abierta",
    "acknowledged": "reconocida",
    "investigating": "en investigacion",
    "closed": "cerrada",
}
ETIQUETAS_DESTINO = {
    "responsable_vigilancia": "Responsable de vigilancia",
    "especialista_radiologia": "Especialista de radiologia",
    "medico_revisor": "Medico de urgencias",
    "responsable_tecnico": "Responsable tecnico",
}
ETIQUETAS_ACTOR = {
    "medico_urgencias": "Medico de urgencias",
    "medico_ap_urgencias": "Medico de urgencias",
    "motor_vigilancia": "Motor de vigilancia",
    "responsable_vigilancia": "Responsable de vigilancia",
}

# Tipos de evento de auditoría
ETIQUETAS_EVENTO = {
    "decision_clinica": "Decision clinica",
    "alerta_generada": "Alerta generada",
    "estado_alerta": "Cambio de estado",
}
ORDEN_SEVERIDAD = {"red": 0, "orange": 1, "yellow": 2}


# =============================================================================
# 1) Columnas esperadas en cada tabla
# =============================================================================

# La app parte de CSVs regenerados por el notebook. En la app NO se usa la
# etiqueta MURA como ground truth operativo. El resultado final de vigilancia
# es la etiqueta normal/anormal que introduce manualmente el medico de urgencias.

# Columnas de feedback_manual.csv
COLUMNAS_FEEDBACK = [
    "id_feedback", "id_estudio", "fecha", "estado_acuerdo", "nota",
    "probabilidad_anormalidad", "etiqueta_predicha", "etiqueta_ap",
    "incidencia_revision", "hash_modelo",
]
# Columnas del registro de inferencias del modelo
COLUMNAS_REGISTRO_INFERENCIAS = [
    "id_estudio", "id_traza", "hash_paciente", "parte_anatomica",
    "ruta_imagen", "ruta_imagen_analizada", "probabilidad_anormalidad",
    "etiqueta_predicha", "tiempo_inferencia_ms", "hash_modelo",
    "fecha_recepcion",
]
# Columnas de alertas_generadas.csv
COLUMNAS_ALERTAS = [
    "id_alerta", "fecha_creacion", "codigo_alerta", "nombre_alerta", "severidad",
    "estado", "indicador", "valor_observado", "umbral", "alcance",
    "tamano_muestra", "evidencias", "accion_recomendada", "destinatario",
    "origen", "patron_emergente", "grupo_afectado", "hash_modelo",
]
# Columnas de alertas_estado.csv
COLUMNAS_ESTADO = [
    "id_alerta", "estado", "fecha_actualizacion", "motivo_cierre", "accion_realizada",
    "responsable", "requiere_seguimiento",
]
# Columnas de audit_log.csv
COLUMNAS_AUDITORIA = [
    "id_evento", "fecha", "tipo_evento", "id_caso", "id_traza",
    "hash_modelo", "version_modelo", "prediccion", "probabilidad_anormalidad",
    "decision_humana", "resultado_final", "alerta_generada",
    "accion_recomendada", "estado_alerta", "id_alerta", "actor", "notas",
    "motivo_cierre", "accion_realizada", "requiere_seguimiento",
]
# Versión reducida de auditría que se muestra en la interfaz
COLUMNAS_AUDITORIA_UI = [
    "id_evento", "fecha", "tipo_evento", "id_caso", "prediccion",
    "probabilidad_anormalidad", "decision_humana", "resultado_final",
    "alerta_generada", "accion_recomendada", "estado_alerta", "id_alerta",
    "actor", "notas", "motivo_cierre", "accion_realizada", "requiere_seguimiento",
]


# =============================================================================
# 2) Lectura de CSVs y rutas
# =============================================================================

# Devuelve la fecha actual UTC con formato ISO simplificado
def fecha_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def leer_csv(path, columns=None, required=False):   # Required para CVS obligatorios
    """Lee un CSV y devuelve siempre un DataFrame con las columnas esperadas."""

    if not path.exists():  
        if required:
            raise FileNotFoundError(path)           # Si era obligatorio y no existe path devuelve error
        return pd.DataFrame(columns=columns or [])  # Si no era obligatorio devuelve DataFrame vacío con las columnas esperadas

    try:                               # Si CSV existe pero vacío, devuelve DataFrame vacío
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=columns or [])

    if columns:                        # Si se pasan las columnas esperadas, las leemos y añadimos NaN para las que falten
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
    return df


def resolver_ruta(valor): 
    """Convierte el texto de una ruta en un Path dentro del repo.

    Acepta rutas ya correctas del repo y tambien rutas antiguas que venian
    del proyecto original dentro de una carpeta `code/` o con rutas absolutas
    del ordenador donde se genero el CSV.
    """

    if pd.isna(valor):
        return None

    texto = str(valor).strip()
    if not texto:
        return None

    ruta = Path(texto)
    candidatos = [ruta]

    if not ruta.is_absolute():
        candidatos.append(DIR_REPO / texto)

    if texto.startswith("code/"):
        candidatos.append(DIR_REPO / texto.split("code/", 1)[1])

    if "/code/" in texto:
        candidatos.append(DIR_REPO / texto.split("/code/", 1)[1])

    if "data/" in texto:
        candidatos.append(DIR_REPO / texto.split("data/", 1)[0] / "data" / texto.split("data/", 1)[1])
        candidatos.append(DIR_DATA / texto.split("data/", 1)[1])

    if "MURA-v1.1/" in texto:
        candidatos.append(DIR_MURA / texto.split("MURA-v1.1/", 1)[1])

    if "outputs/" in texto:
        candidatos.append(DIR_OUTPUTS / texto.split("outputs/", 1)[1])

    for candidato in candidatos:
        if candidato.exists():
            return candidato.resolve()

    if not ruta.is_absolute():
        return DIR_REPO / texto
    return ruta


def archivo_registro_inferencias():
    """Devuelve el CSV de inferencias, aceptando el nombre antiguo si hace falta."""

    if ARCHIVO_REGISTRO_INFERENCIAS.exists():
        return ARCHIVO_REGISTRO_INFERENCIAS
    return ARCHIVO_REGISTRO_INFERENCIAS_ANTIGUO


def cargar_registro_inferencias():
    """Carga el registro de inferencias generado por el notebook."""

    df = leer_csv(archivo_registro_inferencias(), required=True).copy()  # Hacemos copy para evitar modificar el DataFrame original al añadir columnas nuevas
    missing = [col for col in COLUMNAS_REGISTRO_INFERENCIAS if col not in df.columns]  # Miramos si falta alguna columna
    if missing:
        raise ValueError(f"Faltan columnas en el registro de inferencias del modelo: {missing}")
    df = df[COLUMNAS_REGISTRO_INFERENCIAS].copy()   # Nos quedamos solo con las columnas esperadas y en el orden correcto

    # Añadimos columnas vacías
    df["etiqueta_final"] = np.nan
    df["etiqueta_ap"] = np.nan
    df["incidencia_revision"] = ""

    numeric_cols = ["etiqueta_predicha", "etiqueta_final", "etiqueta_ap", "probabilidad_anormalidad", "tiempo_inferencia_ms",]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") # Ponemos las columnas a números,  coerce pone NaN si no se puede convertir, en vez de lanzar error

    df["fecha_recepcion_dt"] = pd.to_datetime(df["fecha_recepcion"], errors="coerce", utc=True)         # Convertimos a datetime de pandas
    df["ruta_imagen_resuelta"] = df["ruta_imagen"].apply(resolver_ruta).astype(str)                     # Resuleve ruta de imagen original
    df["ruta_imagen_analizada_resuelta"] = df["ruta_imagen_analizada"].apply(resolver_ruta).astype(str) # Resuleve ruta de imagen analizada

    return df.sort_values("fecha_recepcion_dt").reset_index(drop=True)


def cargar_feedback():
    return leer_csv(ARCHIVO_FEEDBACK, COLUMNAS_FEEDBACK) # Cargamos el feedback manual con las columnas esperadas


def guardar_feedback(row, etiqueta_ap, incidencia_revision, note):  # Row es la fila del caso revisado
    """Guarda la conclusion del medico de urgencias como ground truth operativo.

    El feedback queda asociado al hash del modelo. Si se reentrena el notebook
    y cambia la version/hash del modelo, la app no reutiliza valoraciones antiguas.
    """

    DIR_TABLAS.mkdir(parents=True, exist_ok=True) # Creamos tabla, parentes=True crea tablas intermedias y exist_ok=True no da error si ya existe
    feedback = cargar_feedback()
    etiqueta_ap = int(etiqueta_ap)
    etiqueta_predicha = int(row.get("etiqueta_predicha"))

    decision = "agree" if etiqueta_predicha == etiqueta_ap else "disagree"
    new_row = {
        "id_feedback": uuid4().hex[:10],  # Id único de 10 caracteres
        "id_estudio": row["id_estudio"],  # Guardamos caso revisado
        "fecha": fecha_utc(),
        "estado_acuerdo": decision,
        "nota": note or "",               # Guardamos la nota o sino vacía
        "probabilidad_anormalidad": row.get("probabilidad_anormalidad"),
        "etiqueta_predicha": row.get("etiqueta_predicha"),
        "etiqueta_ap": etiqueta_ap,
        "incidencia_revision": incidencia_revision or "none",
        "hash_modelo": row.get("hash_modelo", ""),
    }
    # Si el CSV está vacío, evitamos concatenar con un DataFrame vacío para no
    # disparar avisos innecesarios de pandas.
    nuevo_feedback = pd.DataFrame([new_row], columns=COLUMNAS_FEEDBACK)
    if feedback.empty:
        nuevo_feedback.to_csv(ARCHIVO_FEEDBACK, index=False)
    else:
        pd.concat([feedback, nuevo_feedback], ignore_index=True).to_csv(ARCHIVO_FEEDBACK, index=False)
    return new_row


def aplicar_feedback(registro_inferencias, feedback):
    """Une la ultima decision humana disponible con cada inferencia del modelo."""

    # Inicializamos colunas nuevas 
    df = registro_inferencias.copy()
    df["estado_acuerdo_efectivo"] = "not_reviewed"
    df["tiene_feedback_manual"] = False
    df["fecha_feedback"] = ""
    df["incidencia_revision_efectiva"] = ""

    if feedback.empty:
        return df

    # La clave de union incluye hash_modelo para no mezclar revisiones hechas sobre una version anterior del modelo.
    fb = feedback.dropna(subset=["id_estudio"]).copy()  # Eliminamos filas donde id_estudio es NaN, no tiene sentido unir feedback si no se sabe a que caso pertenece
    fb["hash_modelo"] = fb["hash_modelo"].fillna("").astype(str) # Pasamos a str y rellenamos con Nan
    df["hash_modelo"] = df["hash_modelo"].fillna("").astype(str)

    df["_clave_feedback"] = df["id_estudio"].astype(str) + "::" + df["hash_modelo"].astype(str) # Creamos una clave de unión que combina id_estudio y hash_modelo, para luego unir con el feedback
    fb["_clave_feedback"] = fb["id_estudio"].astype(str) + "::" + fb["hash_modelo"].astype(str)

    # Nos quedamos con el feedback más reciente de cada caso ordenado por fecha, agrupado por clave de feedback
    latest = (
        fb.sort_values("fecha")
        .groupby("_clave_feedback", as_index=False)
        .tail(1)
        .set_index("_clave_feedback")
    )
    latest["etiqueta_ap"] = pd.to_numeric(latest["etiqueta_ap"], errors="coerce")
    latest["incidencia_revision"] = latest["incidencia_revision"].fillna("none").replace("", "none")

    mask = df["_clave_feedback"].isin(latest.index)
    df.loc[mask, "tiene_feedback_manual"] = True
    df.loc[mask, "fecha_feedback"] = df.loc[mask, "_clave_feedback"].map(latest["fecha"])
    df.loc[mask, "etiqueta_ap"] = df.loc[mask, "_clave_feedback"].map(latest["etiqueta_ap"])
    # etiqueta_final es la columna interna que usan las metricas. En esta version
    # equivale siempre al ground truth operativo introducido por urgencias.
    df.loc[mask, "etiqueta_final"] = df.loc[mask, "_clave_feedback"].map(latest["etiqueta_ap"])
    df.loc[mask, "incidencia_revision_efectiva"] = df.loc[mask, "_clave_feedback"].map(latest["incidencia_revision"])

    valid_label = mask & df["etiqueta_final"].notna() & df["etiqueta_predicha"].notna()
    df.loc[valid_label, "estado_acuerdo_efectivo"] = np.where(
        df.loc[valid_label, "etiqueta_final"].astype(int).eq(df.loc[valid_label, "etiqueta_predicha"].astype(int)),
        "agree",
        "disagree",
    )
    df = df.drop(columns=["_clave_feedback"])
    return df


def cargar_estado():
    """Devuelve todo lo que necesita la app en cada refresco de la web."""

    # Cargamos feedback, el registro de inferencias con feedback, los casos revisados, las alertas y la auditoria.
    feedback = cargar_feedback()
    registro = aplicar_feedback(cargar_registro_inferencias(), feedback)
    reviewed = registro[registro["tiene_feedback_manual"]].copy()
    return registro, feedback, reviewed, construir_alertas(reviewed), cargar_auditoria()


# =============================================================================
# 3) Metricas de vigilancia
# =============================================================================

def division_segura(a, b):
    """División, devuelve Nan si no se puede dividir"""
    return np.nan if pd.isna(b) or b == 0 else float(a / b)


def distancia_total_variacion(distribucion_referencia, distribucion_actual):
    """Mide cuanto cambia una distribucion respecto a otra en una escala 0-1."""

    indices = distribucion_referencia.index.union(distribucion_actual.index)
    referencia_alineada = distribucion_referencia.reindex(indices, fill_value=0.0)
    actual_alineada = distribucion_actual.reindex(indices, fill_value=0.0)
    return 0.5 * float((referencia_alineada - actual_alineada).abs().sum())


def distribucion_probabilidades(serie_probabilidades):
    """Agrupa probabilidades de anormalidad en rangos sencillos."""

    rangos = pd.cut(
        serie_probabilidades,
        bins=RANGOS_PROBABILIDAD,
        labels=ETIQUETAS_RANGOS_PROBABILIDAD,
        include_lowest=True,
    )
    return rangos.value_counts(normalize=True, sort=False)


def calcular_drift_simple(df, referencia):
    """Compara el subconjunto revisado con el registro base del modelo."""

    if referencia is None or referencia.empty:
        return {
            "drift_parte_anatomica": np.nan,
            "drift_probabilidad": np.nan,
            "drift_prediccion": np.nan,
            "drift_global": np.nan,
        }

    drift_parte_anatomica = np.nan
    drift_probabilidad = np.nan
    drift_prediccion = np.nan

    partes_revisadas = df["parte_anatomica"].dropna().astype(str) if "parte_anatomica" in df else pd.Series(dtype=str)
    partes_referencia = referencia["parte_anatomica"].dropna().astype(str) if "parte_anatomica" in referencia else pd.Series(dtype=str)
    if not partes_revisadas.empty and not partes_referencia.empty and partes_revisadas.nunique() > 1:
        drift_parte_anatomica = distancia_total_variacion(
            partes_referencia.value_counts(normalize=True),
            partes_revisadas.value_counts(normalize=True),
        )

    prob_revisadas = pd.to_numeric(df.get("probabilidad_anormalidad"), errors="coerce").dropna()
    prob_referencia = pd.to_numeric(referencia.get("probabilidad_anormalidad"), errors="coerce").dropna()
    if not prob_revisadas.empty and not prob_referencia.empty:
        drift_probabilidad = distancia_total_variacion(
            distribucion_probabilidades(prob_referencia),
            distribucion_probabilidades(prob_revisadas),
        )

    pred_revisadas = pd.to_numeric(df.get("etiqueta_predicha"), errors="coerce").dropna().astype(int)
    pred_referencia = pd.to_numeric(referencia.get("etiqueta_predicha"), errors="coerce").dropna().astype(int)
    if not pred_revisadas.empty and not pred_referencia.empty:
        drift_prediccion = distancia_total_variacion(
            pred_referencia.value_counts(normalize=True),
            pred_revisadas.value_counts(normalize=True),
        )

    valores_validos = [
        valor for valor in [drift_parte_anatomica, drift_probabilidad, drift_prediccion]
        if not pd.isna(valor)
    ]
    drift_global = float(np.mean(valores_validos)) if valores_validos else np.nan

    return {
        "drift_parte_anatomica": drift_parte_anatomica,
        "drift_probabilidad": drift_probabilidad,
        "drift_prediccion": drift_prediccion,
        "drift_global": drift_global,
    }


def confianza_prediccion(row): 
    """Calcula la confianza del modelo según su propia predicción"""

    prob = float(row.get("probabilidad_anormalidad", np.nan))
    pred = int(row.get("etiqueta_predicha", 0))

    # Si el modelo predice anormal (1), su confianza es la probabilidad de anormalidad y viceversa
    return prob if pred == 1 else 1 - prob


def errores_alta_confianza(df):
    """Buscamos errores donde el modelo estaba muy seguro"""

    # Nos quedamos solo con los casos que tienen estas etiquetas
    data = df.dropna(subset=["etiqueta_final", "etiqueta_predicha", "probabilidad_anormalidad"]).copy()

    if data.empty:
        return data
    data["confianza_prediccion"] = data.apply(confianza_prediccion, axis=1) # Calculamos confianza

    # Devolvemos solo casos donde confianza >= 0.8 y el modelo se ha equivocado
    return data[
        (data["confianza_prediccion"] >= UMBRAL_ALTA_CONFIANZA) & 
        (data["etiqueta_final"].astype(int) != data["etiqueta_predicha"].astype(int))
    ].copy()


def falsos_negativos_alta_confianza(df): 
    """Buscamos casos donde el modelo predice normal con alta confianza pero urgencias dice que es anormal"""

    data = df.dropna(subset=["etiqueta_final", "etiqueta_predicha", "probabilidad_anormalidad"]).copy()

    if data.empty:
        return data
    data["confianza_prediccion"] = data.apply(confianza_prediccion, axis=1)

    return data[
        (data["etiqueta_final"].astype(int) == 1) & 
        (data["etiqueta_predicha"].astype(int) == 0) & 
        (data["confianza_prediccion"] >= UMBRAL_ALTA_CONFIANZA)
    ].copy()


def calcular_metricas(df, referencia=None):
    """Calcula indicadores principales de vigilancia comparando modelo vs ground truth introducido por urgencias."""

    usable = df.dropna(subset=["etiqueta_final", "etiqueta_predicha"]).copy()
    if usable.empty:
        return {
            "n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "recall": np.nan, "fnr": np.nan, "f1": np.nan,
            "accuracy": np.nan, "desacuerdo": np.nan,
            "latencia_p95": np.nan, "fn_alta_conf": 0,
            "error_alta_conf": 0, "error_alta_conf_rate": np.nan,
            "mala_calidad_rate": np.nan,
            "drift_parte_anatomica": np.nan,
            "drift_probabilidad": np.nan,
            "drift_prediccion": np.nan,
            "drift_global": np.nan,
            "n_normal": 0, "n_anormal": 0,
            "acierto_normal": np.nan, "acierto_anormal": np.nan,
            "desacuerdo_normal": np.nan, "desacuerdo_anormal": np.nan,
            "error_alta_conf_rate_normal": np.nan,
            "error_alta_conf_rate_anormal": np.nan,
        }

    if referencia is None:
        referencia = cargar_registro_inferencias() if archivo_registro_inferencias().exists() else df

    etiqueta_ap = usable["etiqueta_final"].astype(int)          # Convertimos a int para asegurarnos de que son 0 y 1
    prediccion_modelo = usable["etiqueta_predicha"].astype(int) # Convertimos a int para asegurarnos de que son 0 y 1

    # Matriz de confusion binaria:
    # - positivo/anormal = 1
    # - negativo/normal = 0
    tp = int(((prediccion_modelo == 1) & (etiqueta_ap == 1)).sum()) # modelo dice anormal y urgencias dice anormal
    tn = int(((prediccion_modelo == 0) & (etiqueta_ap == 0)).sum()) # modelo dice normal y urgencias dice normal
    fp = int(((prediccion_modelo == 1) & (etiqueta_ap == 0)).sum()) # modelo dice anormal pero urgencias dice normal
    fn = int(((prediccion_modelo == 0) & (etiqueta_ap == 1)).sum()) # modelo dice normal pero urgencias dice anormal
    precision = division_segura(tp, tp + fp)
    recall = division_segura(tp, tp + fn)
    estado_col = "estado_acuerdo_efectivo" if "estado_acuerdo_efectivo" in df else "estado_acuerdo"
    errores_confianza_alta = errores_alta_confianza(usable)
    fn_confianza_alta = falsos_negativos_alta_confianza(usable)
    # incidencia_revision_efectiva viene de aplicar_feedback; incidencia_revision viene del feedback guardado.
    columna_incidencia = "incidencia_revision_efectiva" if "incidencia_revision_efectiva" in df else "incidencia_revision"

    # Calculamos métricas separadas para las dos clases, normal (0) y anormal (1)
    metricas_clase = {}
    for label, suffix in [(0, "normal"), (1, "anormal")]:
        subset = usable[etiqueta_ap == label]
        n_label = int(len(subset))

        # Guardamos en el diccionario de métricas por clase si hay casos
        if n_label:
            errores_clase = errores_confianza_alta[errores_confianza_alta["etiqueta_final"].astype(int) == label]
            metricas_clase[f"n_{suffix}"] = n_label
            metricas_clase[f"acierto_{suffix}"] = float((subset["etiqueta_predicha"].astype(int) == label).mean())
            metricas_clase[f"desacuerdo_{suffix}"] = float((subset["etiqueta_predicha"].astype(int) != label).mean())
            metricas_clase[f"error_alta_conf_rate_{suffix}"] = division_segura(len(errores_clase), n_label)
        else:
            metricas_clase[f"n_{suffix}"] = 0
            metricas_clase[f"acierto_{suffix}"] = np.nan
            metricas_clase[f"desacuerdo_{suffix}"] = np.nan
            metricas_clase[f"error_alta_conf_rate_{suffix}"] = np.nan

    drift = calcular_drift_simple(usable, referencia)

    # Devolvemos un diccionario con todas las métricas calculadas, incluyendo las generales y por clase.
    return {
        "n": int(len(usable)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "recall": recall,
        "fnr": division_segura(fn, tp + fn),
        "f1": division_segura(2 * precision * recall, precision + recall),
        "accuracy": division_segura(tp + tn, len(usable)),
        "desacuerdo": float((df[estado_col] == "disagree").mean()) if len(df) else np.nan,
        "latencia_p95": float(df["tiempo_inferencia_ms"].quantile(0.95)) if len(df) else np.nan,
        "fn_alta_conf": int(len(fn_confianza_alta)),
        "error_alta_conf": int(len(errores_confianza_alta)),
        "error_alta_conf_rate": division_segura(len(errores_confianza_alta), len(usable)),
        "mala_calidad_rate": float((df[columna_incidencia] == "poor_image_quality").mean()) if columna_incidencia in df and len(df) else np.nan,
        **drift,
        **metricas_clase, # Desempaquetamos diccionario dentro de otro, es decir, añadimos las métricas por clase al diccionario general
    }


def matriz_confusion(df):
    """Devuelve la matriz urgencias vs modelo en formato largo para Altair."""

    usable = df.dropna(subset=["etiqueta_final", "etiqueta_predicha"]).copy()
    etiquetas = [(0, "normal"), (1, "anormal")]

    if usable.empty:
        matriz = np.zeros((2, 2), dtype=int)
    else:
        matriz = confusion_matrix(
            usable["etiqueta_final"].astype(int),
            usable["etiqueta_predicha"].astype(int),
            labels=[0, 1],
        )
    filas = []
    for fila, (_, nombre_real) in enumerate(etiquetas):
        for columna, (_, nombre_predicho) in enumerate(etiquetas):
            filas.append({
                "referencia": nombre_real,
                "prediccion": nombre_predicho,
                "casos": int(matriz[fila, columna]),
            })
    return pd.DataFrame(filas)


def traza_metricas(reviewed, referencia=None):  # Casos revisados por medico de urgencias
    """Construye la gráfica de evolución acumulada caso a caso."""

    if reviewed.empty:
        return pd.DataFrame()

    if referencia is None:
        referencia = cargar_registro_inferencias() if archivo_registro_inferencias().exists() else reviewed
    
    ordered = reviewed.copy()
    ordered["fecha_feedback_dt"] = pd.to_datetime(ordered["fecha_feedback"], errors="coerce", utc=True) # Convertimos a datetime la fecha feedback
    ordered = ordered.sort_values(["fecha_feedback_dt", "id_estudio"]).reset_index(drop=True)           # Ordenamos por fecha y por id_estudio
    rows = []
    for i in range(1, len(ordered) + 1):  # Recorremos caso a caso acumulado 
        current = ordered.iloc[i - 1]     # Cogemos el caso actual
        rows.append({                     # Calculamos las métricas usando todos los casos revisados hasta el caso actual (inclusive)
            "caso_n": i,
            "id_estudio": current["id_estudio"],
            "latencia_caso_ms": current.get("tiempo_inferencia_ms", np.nan),
            **calcular_metricas(ordered.iloc[:i], referencia),
        })
    return pd.DataFrame(rows)


def metricas_por_parte_anatomica(df, referencia=None):
    """Calculamos las métricas agrupadas por parte anatómica"""

    if referencia is None:
        referencia = cargar_registro_inferencias() if archivo_registro_inferencias().exists() else df

    rows = []
    # Parte_anatomica es la columna de la tabla (codo, muñeca ...) y group es el subconjunto de casos que corresponden a esa parte anatómica.
    for parte_anatomica, group in df.dropna(subset=["parte_anatomica"]).groupby("parte_anatomica"):
        referencia_parte = referencia[referencia["parte_anatomica"] == parte_anatomica] if "parte_anatomica" in referencia else referencia
        rows.append({"parte_anatomica": parte_anatomica, **calcular_metricas(group, referencia_parte)})
    return pd.DataFrame(rows).sort_values(["fnr", "recall"], ascending=[False, True]) if rows else pd.DataFrame()


def tabla_calibracion(df):
    """Permite revisar los casos revisados y detectar patrones de error, distinguir entre aciertos, errores normales yerrores de alta y baja confianza."""

    data = df.dropna(subset=["etiqueta_final", "etiqueta_predicha", "probabilidad_anormalidad"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["rango_probabilidad", "casos", "anormal_real", "pred_anormal", "error", "error_alta_conf"]) # Si no hay datos, tabla vacía con columnas esperadas 

    # Dividimos la probabilidad en rangos
    data["rango_probabilidad"] = pd.cut(
        data["probabilidad_anormalidad"],
        bins=RANGOS_PROBABILIDAD,
        labels=ETIQUETAS_RANGOS_PROBABILIDAD,
        include_lowest=True,
    )
    # Marcamos como error los casos donde la etiqueta final (urgencias) no coincide con la predicha por el modelo, calculamos la confianza de cada prediccion e identificamos errores de alta confianza.
    data["error"] = data["etiqueta_final"].astype(int) != data["etiqueta_predicha"].astype(int)
    data["confianza_prediccion"] = data.apply(confianza_prediccion, axis=1)
    data["error_alta_conf"] = data["error"] & (data["confianza_prediccion"] >= UMBRAL_ALTA_CONFIANZA)

    rows = []
    for label, group in data.groupby("rango_probabilidad", observed=False):
        if not group.empty:
            rows.append({
                "rango_probabilidad": str(label),
                "casos": int(len(group)),
                "anormal_real": float(group["etiqueta_final"].astype(int).mean()),
                "pred_anormal": float(group["etiqueta_predicha"].astype(int).mean()),
                "error": float(group["error"].mean()),
                "error_alta_conf": int(group["error_alta_conf"].sum()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 4) Reglas de alertas emergentes
# =============================================================================

def clasificar_valor_bajo(value, yellow, orange, red): 
    """Clasifica indicadores donde un valor bajo es peor"""

    if pd.isna(value):
        return None
    return "red" if value < red else "orange" if value < orange else "yellow" if value < yellow else None


def clasificar_valor_alto(value, yellow, orange, red): 
    """Clasifica indicadores donde un valor alto es peor"""
    
    if pd.isna(value):
        return None
    return "red" if value > red else "orange" if value > orange else "yellow" if value > yellow else None


def peor_severidad(*severidades): 
    """Devuelve la peor severidad entre varias, siguiendo el orden red > orange > yellow. Si no hay ninguna severidad, devuelve None."""
    
    for severidad in ["red", "orange", "yellow"]:
        if severidad in severidades:
            return severidad
    return None


def crear_alerta(code, name, severidad, indicator, observed, umbral, scope,
                 tamano_muestra, action, destinatario, evidence="",
                 pattern="", grupo_afectado="", hash_modelo=""):
    """Crea alerta como un diccionario con toda la información relevante"""
    
    # El hash entra en el ID para que un reentrenamiento cree un ciclo de alertas nuevo, en lugar de reciclar alertas de un modelo anterior.
    model_fragment = f"_{str(hash_modelo)[:8]}" if hash_modelo else ""
    return {
        "id_alerta": f"alerta_{code.lower()}_{scope.lower()}_{severidad}{model_fragment}".replace(" ", "_"),
        "fecha_creacion": fecha_utc(),
        "codigo_alerta": code,
        "nombre_alerta": name,
        "severidad": severidad,
        "estado": "open",
        "indicador": indicator,
        "valor_observado": round(float(observed), 4),
        "umbral": umbral,
        "alcance": scope,
        "tamano_muestra": tamano_muestra,
        "evidencias": evidence,
        "accion_recomendada": action,
        "destinatario": destinatario,
        "origen": "alerta_emergente",
        "patron_emergente": pattern or name,
        "grupo_afectado": grupo_afectado or scope,
        "hash_modelo": hash_modelo,
    }


def construir_alertas(reviewed):
    """Genera alertas emergentes a partir de reglas simples y trazables."""

    alerts = []
    referencia = cargar_registro_inferencias() if archivo_registro_inferencias().exists() else reviewed
    metrics = calcular_metricas(reviewed, referencia)
    # Si hay menos de 5 casos revisados, no se generan alertas
    if metrics["n"] < MIN_CASOS_ALERTA: 
        return pd.DataFrame(columns=COLUMNAS_ALERTAS)
    # Definimos las primeras alertas son globales
    scope = "global"
    hash_modelo = str(reviewed["hash_modelo"].dropna().astype(str).mode().iat[0]) if "hash_modelo" in reviewed and not reviewed["hash_modelo"].dropna().empty else ""

    # Lista de reglas de alertas.
    # Devuelve alerta y severidad según el valor del indicador observado
    checks = [
        ("CLIN-01", "Patron emergente: recall bajo", clasificar_valor_bajo(metrics["recall"], 0.80, 0.75, 0.70),
         "recall", metrics["recall"], 0.70, "Revisar positivos omitidos y subgrupos afectados", "responsable_vigilancia",
         "", "Descenso de sensibilidad agregado", scope),
        ("CLIN-02", "Patron emergente: FNR elevado", clasificar_valor_alto(metrics["fnr"], 0.18, 0.25, 0.30),
         "false_negative_rate", metrics["fnr"], 0.30, "Listar falsos negativos y revisar calidad/parte anatomica", "responsable_vigilancia",
         "", "Aumento de falsos negativos", scope),
        ("USE-01", "Patron emergente: desacuerdo clinico alto", clasificar_valor_alto(metrics["desacuerdo"], 0.20, 0.30, 0.40),
         "clinical_disagreement_rate", metrics["desacuerdo"], 0.30, "Revisar feedback clinico y presentacion de resultados", "responsable_vigilancia",
         "", "Incremento de discrepancias humano-IA", scope),
        ("TECH-01", "Patron emergente: latencia p95 elevada", clasificar_valor_alto(metrics["latencia_p95"], 1500, 2500, 4000),
         "p95_latency_ms", metrics["latencia_p95"], 2500, "Revisar cola de procesamiento y carga tecnica simulada", "responsable_tecnico",
         "", "Degradacion del tiempo de respuesta", scope),
        ("DATA-01", "Patron emergente: drift de distribucion", clasificar_valor_alto(metrics["drift_global"], 0.15, 0.25, 0.35),
         "drift_distribucion_simple", metrics["drift_global"], 0.25, "Revisar cambios en partes anatomicas, probabilidades y balance de predicciones", "responsable_vigilancia",
         "", "Cambio de distribucion respecto al registro base del modelo", scope),
    ]

    high_conf_fn = falsos_negativos_alta_confianza(reviewed)
    # Si hay alguno, creamos la alerta
    if not high_conf_fn.empty:
        checks.append((
            "CLIN-03", "Patron emergente: anormalidad omitida con alta confianza",
            "red" if len(high_conf_fn) >= 2 else "orange",
            "high_confidence_false_negative_count", float(len(high_conf_fn)), 1.0,
            "Revision prioritaria de casos con probabilidad baja y etiqueta de urgencias anormal",
            "especialista_radiologia", ";".join(high_conf_fn["id_estudio"].astype(str).head(5)),
            "Falsos negativos con confianza alta", scope,
        ))

    high_conf_error = errores_alta_confianza(reviewed)
    # Si hay alguno, creamos la alerta
    if not high_conf_error.empty:
        rate = division_segura(len(high_conf_error), metrics["n"])
        severidad = "red" if len(high_conf_error) >= 3 or rate >= 0.25 else "orange" if len(high_conf_error) >= 2 else "yellow"
        checks.append((
            "CAL-01", "Patron emergente: error de alta confianza recurrente", severidad,
            "high_confidence_error_count", float(len(high_conf_error)), 1.0,
            "Auditar calibracion del modelo y revisar errores de alta confianza",
            "responsable_vigilancia", ";".join(high_conf_error["id_estudio"].astype(str).head(5)),
            "Predicciones erroneas con confianza >= 0.80", scope,
        ))

    for code, name, severidad, indicator, observed, umbral, action, dest, evidence, pattern, group_name in checks:
        # Solo se crea la alerta si se ha devuelto severidad
        if severidad:
            alerts.append(crear_alerta(
                code, name, severidad, indicator, observed, umbral, scope,
                metrics["n"], action, dest, evidence, pattern, group_name,
                hash_modelo,
            ))
    # Añadimos alertas por parte anatómica
    alerts.extend(construir_alertas_por_parte(reviewed, referencia, hash_modelo))
    # Devolvemos DataFrame de alertas
    return pd.DataFrame(alerts, columns=COLUMNAS_ALERTAS) if alerts else pd.DataFrame(columns=COLUMNAS_ALERTAS)


def construir_alertas_por_parte(group, referencia, hash_modelo=""):
    """Busca degradaciones localizadas en una region anatomica concreta."""

    alerts = []
    # Agrupa por parte anatómica y calcula métricas para cada grupo
    for parte_anatomica, body_group in group.dropna(subset=["parte_anatomica"]).groupby("parte_anatomica"):
        referencia_parte = referencia[referencia["parte_anatomica"] == parte_anatomica] if "parte_anatomica" in referencia else referencia
        metrics = calcular_metricas(body_group, referencia_parte)
        if metrics["n"] < MIN_CASOS_ALERTA:
            continue
        severidad = peor_severidad(
            clasificar_valor_bajo(metrics["recall"], 0.80, 0.75, 0.70),
            clasificar_valor_alto(metrics["fnr"], 0.18, 0.25, 0.30),
        )
        if not severidad:
            continue

        valid_cases = body_group.dropna(subset=["etiqueta_final", "etiqueta_predicha"]).copy()
        false_negatives = valid_cases[
            (valid_cases["etiqueta_final"].astype(int) == 1) & 
            (valid_cases["etiqueta_predicha"].astype(int) == 0)
        ]
        # Alerta de degradación localizada en parte anatómica concreta
        alerts.append(crear_alerta(
            "CLIN-04",
            f"Patron emergente: degradacion localizada en {parte_anatomica}",
            severidad,
            "localized_false_negative_rate",
            metrics["fnr"],
            0.18,
            parte_anatomica,
            metrics["n"],
            f"Revisar rendimiento localizado en {parte_anatomica}; comparar tecnica, calidad y distribucion",
            "responsable_vigilancia",
            ";".join(false_negatives["id_estudio"].astype(str).head(5)),
            f"Rendimiento inferior al umbral en subgrupo anatomico {parte_anatomica}",
            parte_anatomica,
            hash_modelo,
        ))
    return alerts


def cargar_alertas_previas(): # Cargamos alertas previamente guardadas
    return leer_csv(ARCHIVO_ALERTAS, COLUMNAS_ALERTAS)


def guardar_alertas_previas(alerts): # Guarda alertas en CSV
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    alerts.reindex(columns=COLUMNAS_ALERTAS).to_csv(ARCHIVO_ALERTAS, index=False)


def alertas_nuevas(previous, current):
    """Compara alertas anteriores con actuales"""

    old_ids = set(previous.get("id_alerta", pd.Series(dtype=str)).astype(str)) # Ids antiguos
    current_ids = current["id_alerta"].astype(str)                             # Ids actuales
    mask_new = ~current_ids.isin(old_ids)
    return current[mask_new].to_dict("records") if not current.empty else []


def cargar_estado_alertas(id_alertas=None): 
    state = leer_csv(ARCHIVO_ESTADO_ALERTAS, COLUMNAS_ESTADO)
    if not id_alertas:
        return state

    known = set(state["id_alerta"].astype(str))
    missing = [id_alerta for id_alerta in id_alertas if str(id_alerta) not in known]
    if missing:
        new_rows = []
        for id_alerta in missing:
            new_rows.append({
                "id_alerta": id_alerta, "estado": "open", "fecha_actualizacion": "",
                "motivo_cierre": "", "accion_realizada": "", "responsable": "",
                "requiere_seguimiento": "",
            })
        state = pd.concat([state, pd.DataFrame(new_rows)], ignore_index=True)
        state.reindex(columns=COLUMNAS_ESTADO).to_csv(ARCHIVO_ESTADO_ALERTAS, index=False)
    return state


def combinar_estado_alertas(alerts):
    if alerts.empty:
        return alerts.copy()
    state = cargar_estado_alertas(alerts["id_alerta"].astype(str).tolist())
    merged = alerts.merge(state, on="id_alerta", how="left", suffixes=("", "_guardado"))
    merged["estado_efectivo"] = merged["estado_guardado"].fillna(merged["estado"])
    return merged


def cambiar_estado_alerta(id_alerta, estado, motivo_cierre="",
                          accion_realizada="", responsable="",
                          requiere_seguimiento=""):
    """Actualiza el ciclo de vida de una alerta sin borrar su historial."""
    state = cargar_estado_alertas([id_alerta])
    mask = state["id_alerta"].astype(str) == str(id_alerta)
    state.loc[mask, "estado"] = estado
    state.loc[mask, "fecha_actualizacion"] = fecha_utc()
    if estado == "closed":
        state.loc[mask, "motivo_cierre"] = motivo_cierre
        state.loc[mask, "accion_realizada"] = accion_realizada
        state.loc[mask, "responsable"] = responsable
        state.loc[mask, "requiere_seguimiento"] = requiere_seguimiento
    state.reindex(columns=COLUMNAS_ESTADO).to_csv(ARCHIVO_ESTADO_ALERTAS, index=False)


# =============================================================================
# 5) Auditoria e informe exportable
# =============================================================================

def cargar_auditoria():
    return leer_csv(ARCHIVO_AUDITORIA, COLUMNAS_AUDITORIA)


def guardar_eventos_auditoria(events):
    """Añade eventos al registro de auditoria conservando los anteriores."""

    if not events:
        return
    audit = cargar_auditoria()
    rows = []
    for event in events:
        row = {col: event.get(col, "") for col in COLUMNAS_AUDITORIA}
        row["id_evento"] = row["id_evento"] or "audit_" + uuid4().hex[:10]
        row["fecha"] = row["fecha"] or fecha_utc()
        rows.append(row)
    nuevos_eventos = pd.DataFrame(rows, columns=COLUMNAS_AUDITORIA)
    if audit.empty:
        nuevos_eventos.to_csv(ARCHIVO_AUDITORIA, index=False)
    else:
        pd.concat([audit, nuevos_eventos], ignore_index=True).to_csv(ARCHIVO_AUDITORIA, index=False)


def texto_etiqueta(value):
    """Convierte etiqueta numérica a texto"""
    if pd.isna(value):
        return ""
    if str(value) in ["1", "1.0", "anormal"]:
        return "anormal"
    if str(value) in ["0", "0.0", "normal"]:
        return "normal"
    return str(value)


def evento_decision(row, feedback_row, generated_alerts):
    """Construye un evento de auditoría cuando el médico guarda una decisión"""

    issue = feedback_row.get("incidencia_revision", "none")
    note = feedback_row.get("nota", "")
    issue_note = f"Incidencia revision: {ETIQUETAS_REVISION.get(str(issue), issue)}"
    notas = issue_note if not str(note).strip() else f"{issue_note} | {note}"
    return {
        "tipo_evento": "decision_clinica",
        "fecha": feedback_row["fecha"],
        "id_caso": row.get("id_estudio", ""),
        "id_traza": row.get("id_traza", ""),
        "hash_modelo": row.get("hash_modelo", ""),
        "version_modelo": VERSION_MODELO,
        "prediccion": f"{texto_etiqueta(row.get('etiqueta_predicha'))} ({float(row.get('probabilidad_anormalidad')):.3f})",
        "probabilidad_anormalidad": row.get("probabilidad_anormalidad", ""),
        "decision_humana": feedback_row["estado_acuerdo"],
        "resultado_final": texto_etiqueta(feedback_row.get("etiqueta_ap", "")),
        "alerta_generada": ", ".join(sorted({f"{a['codigo_alerta']}:{a['severidad']}" for a in generated_alerts})),
        "accion_recomendada": " | ".join(sorted({a["accion_recomendada"] for a in generated_alerts if a.get("accion_recomendada")})),
        "estado_alerta": "open" if generated_alerts else "",
        "id_alerta": ", ".join(a["id_alerta"] for a in generated_alerts),
        "actor": "medico_urgencias",
        "notas": notas,
    }


def eventos_alertas(alerts):
    """Construye un evento de auditoría cuando se genera un alerta nueva"""
    return [{
        "tipo_evento": "alerta_generada",
        "fecha": alert.get("fecha_creacion", fecha_utc()),
        "id_caso": alert.get("evidencias", ""),
        "hash_modelo": alert.get("hash_modelo", ""),
        "version_modelo": VERSION_MODELO,
        "alerta_generada": f"{alert.get('codigo_alerta', '')}:{alert.get('severidad', '')}",
        "accion_recomendada": alert.get("accion_recomendada", ""),
        "estado_alerta": alert.get("estado", "open"),
        "id_alerta": alert.get("id_alerta", ""),
        "actor": "motor_vigilancia",
        "notas": f"{alert.get('nombre_alerta', '')} | {alert.get('alcance', '')}",
    } for alert in alerts]


def evento_estado_alerta(alert, estado, motivo_cierre="", accion_realizada="", requiere_seguimiento=""):
    """Construye un evento de auditoría cuando se cambia el estado de una alerta"""
    return {
        "tipo_evento": "estado_alerta",
        "id_caso": alert.get("evidencias", ""),
        "hash_modelo": alert.get("hash_modelo", ""),
        "version_modelo": VERSION_MODELO,
        "alerta_generada": f"{alert.get('codigo_alerta', '')}:{alert.get('severidad', '')}",
        "accion_recomendada": alert.get("accion_recomendada", ""),
        "estado_alerta": estado,
        "id_alerta": alert.get("id_alerta", ""),
        "actor": "responsable_vigilancia",
        "notas": alert.get("nombre_alerta", ""),
        "motivo_cierre": motivo_cierre,
        "accion_realizada": accion_realizada,
        "requiere_seguimiento": requiere_seguimiento,
    }


def formatear_metrica(value, pattern):
    """Formatea métricas para el informe, si el valor está vacío devuelve NA"""
    return "NA" if pd.isna(value) else pattern.format(value)


def tabla_markdown(df, columns, max_rows=30):
    """Convierte un DataFrame en una tabla markdown."""
    if df.empty:
        return "Sin datos disponibles.\n"
    use_cols = [col for col in columns if col in df.columns]
    rows = df[use_cols].fillna("").astype(str).head(max_rows).values.tolist()
    header = "| " + " | ".join(use_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(use_cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    suffix = "\n\n_Se muestran las primeras 30 filas._" if len(df) > max_rows else ""
    return "\n".join([header, separator, *body]) + suffix


def construir_informe_vigilancia(registro_inferencias, reviewed, alerts, audit):
    """Crea un informe markdown resumido para documentar la vigilancia."""
    metrics = calcular_metricas(reviewed, registro_inferencias)
    anatomy = metricas_por_parte_anatomica(reviewed, registro_inferencias)
    calibration = tabla_calibracion(reviewed)
    alerts_with_state = combinar_estado_alertas(alerts)
    open_alerts = alerts_with_state[alerts_with_state["estado_efectivo"].astype(str).eq("open")] if not alerts_with_state.empty else alerts_with_state
    coverage = division_segura(len(reviewed), len(registro_inferencias))

    return f"""# Informe de vigilancia poscomercializacion

    Fecha de generacion: {fecha_utc()}

    ## Alcance

    - Sistema de IA: {VERSION_MODELO}
    - Imagenes de referencia: MURA-v1.1 valid
    - Ground truth operativo: valoracion del medico de urgencias introducida en la app
    - Inferencias registradas: {len(registro_inferencias)}
    - Casos revisados por urgencias: {len(reviewed)}
    - Cobertura de revision: {formatear_metrica(coverage, "{:.1%}")}
    - Eventos de auditoria: {len(audit)}

    ## Indicadores principales

    - Recall: {formatear_metrica(metrics["recall"], "{:.3f}")}
    - FNR: {formatear_metrica(metrics["fnr"], "{:.3f}")}
    - F1: {formatear_metrica(metrics["f1"], "{:.3f}")}
    - Accuracy: {formatear_metrica(metrics["accuracy"], "{:.3f}")}
    - Desacuerdo clinico: {formatear_metrica(metrics["desacuerdo"], "{:.1%}")}
    - Latencia p95: {formatear_metrica(metrics["latencia_p95"], "{:.0f} ms")}
    - Errores de alta confianza: {metrics["error_alta_conf"]}
    - Drift global simple: {formatear_metrica(metrics["drift_global"], "{:.3f}")}

    ## Alertas emergentes

    - Total de alertas: {len(alerts_with_state)}
    - Alertas abiertas: {len(open_alerts)}
    - Rojas: {int((alerts_with_state["severidad"] == "red").sum()) if not alerts_with_state.empty else 0}
    - Naranjas: {int((alerts_with_state["severidad"] == "orange").sum()) if not alerts_with_state.empty else 0}
    - Amarillas: {int((alerts_with_state["severidad"] == "yellow").sum()) if not alerts_with_state.empty else 0}

    {tabla_markdown(alerts_with_state, ["codigo_alerta", "severidad", "estado_efectivo", "alcance", "indicador", "valor_observado", "accion_recomendada"])}

    ## Indicadores por parte anatomica

    {tabla_markdown(anatomy, ["parte_anatomica", "n", "recall", "fnr", "accuracy", "desacuerdo", "fn_alta_conf", "error_alta_conf", "drift_global"])}

    ## Calibracion simple

    {tabla_markdown(calibration, ["rango_probabilidad", "casos", "anormal_real", "pred_anormal", "error", "error_alta_conf"])}

    ## Interpretacion

    Este informe documenta una simulacion academica de vigilancia poscomercializacion. El objetivo no es validar clinicamente un producto real, sino demostrar monitorizacion, supervision humana, deteccion de patrones emergentes, gestion de alertas y trazabilidad. En esta vigilancia, las metricas se calculan comparando la prediccion del modelo con el ground truth operativo introducido por urgencias.
    """
