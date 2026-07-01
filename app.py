"""
Interfaz Streamlit.

La logica importante vive en vigilance_backend.py. Este archivo solo muestra el
flujo de revision, las metricas, las alertas y la auditoria de forma sencilla.
"""

import pandas as pd
import streamlit as st 
from datetime import datetime
import vigilance_backend as core

import altair as alt            # Librería de gráficas
from PIL import Image, ImageOps # Image para cargar imágenes y ImageOps aplicar operaciones como contraste

st.set_page_config(page_title="Sistema TFG vigilancia", layout="wide")

COLOR_SEVERIDAD = {
    "red": "#ff4b4b",
    "orange": "#f59e0b",
    "yellow": "#facc15",
}


# =============================================================================
# 1) Utilidades visuales
# =============================================================================

def cargar_imagen_mostrar(ruta):
    """Prepara la radiografia para mostrarla en Streamlit."""
    imagen = Image.open(ruta)
    if imagen.mode in {"L", "I", "I;16", "F"}: # Si la imagen está en alguno de estos modos, aplicamos autocontraste y la convertimos a RGB
        return ImageOps.autocontrast(imagen.convert("L")).convert("RGB") # Convertir a L (escala de grises), aplicar autocontraste y convertir a RGB
    return imagen.convert("RGB")


def hash_corto(valor):
    """Devuelve un hash de 12 primeros caracteres para no llenar la interfaz."""
    if pd.isna(valor) or not str(valor).strip():
        return "sin hash"
    return str(valor)[:12]


def tabla_calibracion_visible(calibracion): # Pestaña de Vigilancia
    """Convierte la tabla de calibración tecnica en una tabla entendible."""
    filas = []
    for _, fila in calibracion.iterrows():
        casos = int(fila.get("casos", 0))  # casos revisados en ese rango de probabilidad
        errores = int(round(float(fila.get("error", 0)) * casos)) if casos else 0
        filas.append({
            "Rango de P(anormalidad)": fila.get("rango_probabilidad", ""),
            "Casos": casos,
            "Anormales segun urgencias": core.formatear_metrica(fila.get("anormal_real"), "{:.1%}"),
            "Predichos anormales": core.formatear_metrica(fila.get("pred_anormal"), "{:.1%}"),
            "Errores": f"{errores}/{casos}",
            "Tasa de error": core.formatear_metrica(fila.get("error"), "{:.1%}"),
            "Errores alta confianza": int(fila.get("error_alta_conf", 0)),
        })
    return pd.DataFrame(filas)


def tabla_auditoria_visible(auditoria):  # Pestaña de Auditoria
    """Prepara la tabla auditoría para leerla como registro historico del sistema."""
    tabla = auditoria.copy()
    if tabla.empty:
        return tabla

    # Cambiamos los valores que toman cada columna, pasando de códigos a etiquetas legibles, p ej. decision_clinica -> Decisión clínica
    if "tipo_evento" in tabla.columns:
        tabla["tipo_evento"] = tabla["tipo_evento"].replace(core.ETIQUETAS_EVENTO)
    if "estado_alerta" in tabla.columns:
        tabla["estado_alerta"] = tabla["estado_alerta"].replace(core.ETIQUETAS_ESTADO)
    if "actor" in tabla.columns:
        tabla["actor"] = tabla["actor"].replace(core.ETIQUETAS_ACTOR)
    if "decision_humana" in tabla.columns:
        tabla["decision_humana"] = tabla["decision_humana"].replace(core.ETIQUETAS_DECISION)
    if "probabilidad_anormalidad" in tabla.columns:
        tabla["probabilidad_anormalidad"] = tabla["probabilidad_anormalidad"].map(
            lambda valor: core.formatear_metrica(valor, "{:.3f}") if str(valor).strip() else ""
        )
    # Cambiamos nombres de las columnas para mostrar la tabla
    columnas = [col for col in core.COLUMNAS_AUDITORIA_UI if col in tabla.columns]
    nombres = {
        "id_evento": "ID evento",
        "fecha": "Fecha",
        "tipo_evento": "Tipo evento",
        "id_caso": "Caso/evidencia",
        "prediccion": "Prediccion modelo",
        "probabilidad_anormalidad": "P(anormalidad)",
        "decision_humana": "Decision humana",
        "resultado_final": "Resultado final",
        "alerta_generada": "Alerta",
        "accion_recomendada": "Accion recomendada",
        "estado_alerta": "Estado en ese evento",
        "id_alerta": "ID alerta",
        "actor": "Actor/origen",
        "notas": "Notas",
        "motivo_cierre": "Motivo cierre",
        "accion_realizada": "Accion realizada",
        "requiere_seguimiento": "Seguimiento",
    }
    return tabla[columnas].rename(columns=nombres)


def mostrar_indicadores_principales(metricas): # Gráfico de vigilancia
    """Muestra los indicadores principales en una misma fila."""

    indicadores = [
        ("Recall", "recall", "{:.3f}"),
        ("FNR", "fnr", "{:.3f}"),
        ("F1", "f1", "{:.3f}"),
        ("Accuracy", "accuracy", "{:.3f}"),
        ("Desacuerdo", "desacuerdo", "{:.1%}"),
        ("Lat. p95", "latencia_p95", "{:.0f} ms"),
    ]
    # Creamos 6 columnas en Streamlit y emparejamos cada columna con una métrica
    for columna, (etiqueta, clave, formato) in zip(st.columns(6), indicadores): 
        _ = columna.metric(etiqueta, core.formatear_metrica(metricas.get(clave), formato))


def texto_etiqueta_ap(valor):
    """Convierte el valor de la etiqueta de urgencias a texto."""
    if pd.isna(valor):
        return "pendiente"
    return "anormal" if int(valor) == 1 else "normal"


def mostrar_matriz_confusion(datos):
    """Pinta normal/anormal comparando urgencias contra la prediccion del modelo."""

    matriz = core.matriz_confusion(datos)
    if matriz.empty:
        st.info("No hay datos suficientes para la matriz de confusion.")
        return
    
    grafico = (
        alt.Chart(matriz)                              # Creamos gráfico usando la tabla matriz
        .mark_rect(stroke="#111827", strokeWidth=1)  # Usamos rectángulos para representar cada celda
        .encode(                                       # Definimos como se codifican los datos visualmente
            x=alt.X("prediccion:N", title="Prediccion", sort=["normal", "anormal"]),  # Predicción del modelo
            y=alt.Y("referencia:N", title="Referencia", sort=["normal", "anormal"]),  # Referencia urgencias
            color=alt.Color("casos:Q", legend=None, scale=alt.Scale(scheme="blues")), # El color depende del número de casos, Q significa cuantitativo
            tooltip=["referencia:N", "prediccion:N", "casos:Q"],                      # Al pasar el ratón muestra referencia, predicción y casos
        )
        .properties(width=380, height=380)                                            # Tamaño del gráfico
    )
    texto_celdas = grafico.mark_text(fontSize=18, fontWeight="bold").encode( # Texto encima de cada celda con el número de casos
        text="casos:Q",
        color=alt.value("white"),
    ) 
    st.altair_chart(grafico + texto_celdas, use_container_width=False)       # Streamlit muestra el gráfico combinado (rectángulos + texto)
 

def grafica_lineas_acotada(datos, columnas_y, titulo, titulo_y, max_y):
    """Pinta graficas de líneas acumuladas evitando escalas negativas."""
    
    columnas = [col for col in columnas_y if col in datos.columns]
    datos_grafica = (
        datos[["caso_n", *columnas]]                                      # Selecciona el caso_n y las columnas de métricas
        .melt(id_vars="caso_n", var_name="indicador", value_name="valor") # Convierte de formato ancho a largo para Altair, con columnas caso_n, indicador y valor
        .dropna(subset=["valor"])                                         # Elimina valores vacíos
    )
    if datos_grafica.empty:
        st.info("Aun no hay valores definidos para esta grafica.")
        return

    nombres = {
        "latencia_caso_ms": "latencia del caso",
        "latencia_p95": "latencia p95 acumulada",
        "error_alta_conf_rate": "error alta confianza",
        "drift_global": "drift global",
        "drift_probabilidad": "drift probabilidad",
        "drift_prediccion": "drift prediccion",
        "drift_parte_anatomica": "drift parte anatomica",
        "acierto_normal": "acierto urgencias normal",
        "acierto_anormal": "acierto urgencias anormal",
        "desacuerdo_normal": "desacuerdo urgencias normal",
        "desacuerdo_anormal": "desacuerdo urgencias anormal",
        "error_alta_conf_rate_normal": "error alta conf. urgencias normal",
        "error_alta_conf_rate_anormal": "error alta conf. urgencias anormal",
    }

    # Reemplazamos los nombres técnicos de las columnas por nombres más legibles para mostrar en la gráfica
    datos_grafica["indicador"] = datos_grafica["indicador"].replace(nombres)
    max_caso = max(1, int(datos["caso_n"].max()))

    # Creamos la gráfica de líneas usando Altair
    st.markdown(f"**{titulo}**")
    grafico = (
        alt.Chart(datos_grafica)
        .mark_line(point=True)
        .encode(
            x=alt.X("caso_n:Q", title="Caso revisado", scale=alt.Scale(domain=[1, max_caso], nice=False)),
            y=alt.Y("valor:Q", title=titulo_y, scale=alt.Scale(domain=[0, max_y], nice=False)),
            color=alt.Color("indicador:N", title="Indicador"),
            tooltip=["caso_n:Q", "indicador:N", alt.Tooltip("valor:Q", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(grafico, use_container_width=True)


def filtrar_casos(datos, prefijo, solo_pendientes=False):
    """Permite aplicar filtros comunes a los casos tocar el dataframe original (selector)."""
    col_parte, col_pendientes = st.columns([3, 1])

    # Se crea un desplegable para filtrar por parte antómica
    parte_anatomica = col_parte.selectbox("Parte anatomica", ["Todas"] + sorted(datos["parte_anatomica"].dropna().unique()), key=f"{prefijo}_body")
    solo_pendientes = col_pendientes.checkbox("Solo pendientes", value=solo_pendientes, key=f"{prefijo}_pending")

    vista = datos.copy()

    # Se muestra la parte concreta elegida
    if parte_anatomica != "Todas":
        vista = vista[vista["parte_anatomica"] == parte_anatomica]

    # Si se elige mostrar solo pendientes, se filtra para mostrar solo los casos que no tienen feedback manual (no revisados)
    if solo_pendientes and "tiene_feedback_manual" in vista:
        vista = vista[~vista["tiene_feedback_manual"]]
    return vista.reset_index(drop=True)



# =============================================================================
# INTERFAZ: Pestaña 1 - Revisión clínica
# =============================================================================

def mostrar_revision(registro_inferencias, valoraciones):
    """Pantalla donde el revisor mira el caso y guarda su valoracion."""

    st.header("Revision clinica")
    st.caption(
        "El medico de urgencias introduce el ground truth operativo. "
        "Las metricas de vigilancia comparan esa conclusion con la prediccion del modelo."
    )

    # Mostrar solo los casos pendientes
    vista = filtrar_casos(registro_inferencias, "review", solo_pendientes=True)
    if vista.empty:
        st.info("No hay casos con esos filtros.")
        return

    # Guarda en sesión que caso se está viendo
    st.session_state.review_pos = min(st.session_state.get("review_pos", 0), len(vista) - 1)

    # Crea tres columnas, botón, barra, botón, para navegar entre casos y mostrar progreso
    boton_anterior, barra, boton_siguiente = st.columns([1, 4, 1])
    if boton_anterior.button("Anterior", use_container_width=True):  # Si se pulsa el botón "Anterior", se resta 1 a la posición actual, pero no puede ser menor que 0
        st.session_state.review_pos = max(0, st.session_state.review_pos - 1)
        st.rerun()
    if boton_siguiente.button("Siguiente", use_container_width=True): # Si se pulsa el botón "Siguiente", se suma 1 a la posición actual, pero no puede ser mayor que el último índice de la vista
        st.session_state.review_pos = min(len(vista) - 1, st.session_state.review_pos + 1)
        st.rerun()

    # Calcula el número de casos revisados para mostrar el progreso en la barra. La posición global se calcula sumando el número de casos revisados, la posición actual en la revisión y 1
    revisados = int(registro_inferencias["tiene_feedback_manual"].sum())
    total_casos = len(registro_inferencias)
    posicion_global = min(total_casos, revisados + st.session_state.review_pos + 1)
    _ = barra.progress(posicion_global / total_casos)
    _ = barra.caption(f"Revision {posicion_global} de {total_casos}")

    # Muestra el caso actual con sus imágenes, información y formulario de valoración. El caso se obtiene de la vista filtrada usando la posición actual en sesión.
    caso = vista.iloc[st.session_state.review_pos]
    st.subheader(str(caso["id_estudio"]))
    col_original, col_analizada, col_info = st.columns([1, 1, 1.2])
    for columna, columna_ruta, titulo_imagen in [
        (col_original, "ruta_imagen_resuelta", "Radiografia original"),
        (col_analizada, "ruta_imagen_analizada_resuelta", "Salida analizada"),
    ]:
        ruta = core.resolver_ruta(caso[columna_ruta])
        if ruta and ruta.exists():
            _ = columna.image(cargar_imagen_mostrar(ruta), caption=titulo_imagen, use_container_width=True)
        else:
            if titulo_imagen == "Radiografia original":
                _ = columna.warning(
                    "Radiografia original no disponible. Para mostrarla debe descargarse "
                    "MURA-v1.1 en `data/MURA-v1.1/`."
                )
            else:
                _ = columna.warning("Imagen no encontrada.")

    # Muestra la información del caso y el formulario para introducir la valoración.
    with col_info:
        st.markdown(f"**Parte anatomica:** {str(caso['parte_anatomica']).replace('XR_', '')}")
        st.markdown(f"**Prediccion:** {core.texto_etiqueta(caso['etiqueta_predicha'])}")
        st.markdown(f"**Probabilidad anormalidad:** {float(caso['probabilidad_anormalidad']):.3f}")
        st.markdown(f"**Ground truth urgencias:** {texto_etiqueta_ap(caso.get('etiqueta_final'))}")
        st.markdown(f"**Modelo:** {hash_corto(caso.get('hash_modelo', ''))}")
        st.markdown(f"**Estado:** {core.ETIQUETAS_DECISION.get(str(caso['estado_acuerdo_efectivo']), caso['estado_acuerdo_efectivo'])}")

    # Formulario para introducir la valoración del caso
    with st.form(f"feedback_{caso['id_estudio']}"):
        etiqueta_ap = st.selectbox(
            "Ground truth operativo del medico de urgencias",
            ["Seleccionar", "normal", "anormal"],
            format_func=lambda valor: "Seleccionar" if valor == "Seleccionar" else ("Normal" if valor == "normal" else "Anormal"),
        )
        incidencia_revision = st.selectbox(
            "Incidencia de revision",
            ["none", "doubtful", "poor_image_quality", "out_of_scope_finding"],
            format_func=lambda valor: core.ETIQUETAS_REVISION[valor],
        )
        nota = st.text_area("Nota libre", height=80)
        guardar_valoracion = st.form_submit_button("Guardar valoracion", type="primary")

    if guardar_valoracion:
        if etiqueta_ap == "Seleccionar":
            st.warning("Selecciona si el medico de urgencias considera el caso normal o anormal.")
        else:
            alertas_anteriores = core.cargar_alertas_previas()
            etiqueta_ap_num = 1 if etiqueta_ap == "anormal" else 0
            valoracion_guardada = core.guardar_feedback(caso, etiqueta_ap_num, incidencia_revision, nota)
            registro_actualizado = core.aplicar_feedback(core.cargar_registro_inferencias(), core.cargar_feedback())
            alertas_actuales = core.construir_alertas(registro_actualizado[registro_actualizado["tiene_feedback_manual"]].copy())
            alertas_nuevas = core.alertas_nuevas(alertas_anteriores, alertas_actuales)
            core.guardar_alertas_previas(alertas_actuales)
            core.guardar_eventos_auditoria([core.evento_decision(caso, valoracion_guardada, alertas_nuevas)] + core.eventos_alertas(alertas_nuevas))
            for alerta in alertas_nuevas:
                st.toast(f"{alerta['codigo_alerta']} - {alerta['nombre_alerta']}")
            st.rerun()

    # El historial se filtra por hash para no mezclar revisiones hechas con diferentes versiones del modelo entrenado en el notebook.
    hash_valoraciones = valoraciones["hash_modelo"].fillna("").astype(str)
    historial = valoraciones[
        (valoraciones["id_estudio"].astype(str) == str(caso["id_estudio"]))
        & (hash_valoraciones == str(caso.get("hash_modelo", "")))
    ]

    # Guardamos en sesión el historial del caso para mostrarlo en la pestaña de auditoría
    if not historial.empty:
        columnas_historial = [
            "fecha", "etiqueta_ap", "estado_acuerdo", "incidencia_revision", "nota",
        ]
        historial_visible = historial[[col for col in columnas_historial if col in historial.columns]].copy()
        if "etiqueta_ap" in historial_visible:
            historial_visible["etiqueta_ap"] = historial_visible["etiqueta_ap"].map(texto_etiqueta_ap)
        if "estado_acuerdo" in historial_visible:
            historial_visible["estado_acuerdo"] = historial_visible["estado_acuerdo"].replace(core.ETIQUETAS_DECISION)
        if "incidencia_revision" in historial_visible:
            historial_visible["incidencia_revision"] = historial_visible["incidencia_revision"].replace(core.ETIQUETAS_REVISION)
        st.dataframe(historial_visible, hide_index=True, use_container_width=True)


# =============================================================================
# INTERFAZ: Pestaña 2 - Vigilancia
# =============================================================================

def mostrar_vigilancia(registro_inferencias, revisados, alertas, auditoria):
    """Pantalla de seguimiento: metricas, calibracion, subgrupos e informe."""

    # st.header escribe el titulo principal de esta pestana.
    st.header("Motor de vigilancia")

    # Si aun no hay revisiones guardadas, no tiene sentido calcular metricas.
    # Streamlit pinta el mensaje con st.info y la funcion termina con return.
    if revisados.empty:
        st.info("Guarda casos revisados para activar metricas y alertas.")
        return

    # st.columns divide la pantalla en tres zonas horizontales.
    # Cada col.metric pinta una tarjeta numerica pequena.
    col_revisados, col_pendientes, col_alertas = st.columns(3)
    _ = col_revisados.metric("Casos revisados", len(revisados))
    _ = col_pendientes.metric("Pendientes", len(registro_inferencias) - len(revisados))
    _ = col_alertas.metric("Alertas emergentes", len(alertas))

    # Los indicadores grandes se calculan en el backend y aqui solo se muestran.
    mostrar_indicadores_principales(core.calcular_metricas(revisados, registro_inferencias))

    # Aviso metodologico: con pocos casos, las metricas cambian mucho caso a caso.
    if len(revisados) < 10:
        st.warning("Muestra pequena: con menos de 10 casos revisados las metricas pueden oscilar mucho.")

    # Esta seccion muestra como evolucionan las metricas cada vez que se guarda un caso.
    st.subheader("Evolucion acumulada")
    st.caption(
        "Cada punto representa el estado del sistema despues de guardar un caso. "
        "La comparacion se hace contra el ground truth operativo introducido por urgencias. "
        "Si una linea empieza mas tarde, esa metrica aun no estaba definida "
        "(por ejemplo, recall/FNR necesitan casos anormales segun urgencias). "
        "La latencia del caso es el tiempo de inferencia de ese estudio; la latencia p95 acumulada "
        "resume el peor 5% aproximado de tiempos observados hasta ese momento. "
        "El drift simple mide si los casos revisados se estan alejando del registro base en partes anatomicas, "
        "probabilidades y balance de predicciones."
    )

    # traza_metricas reconstruye la serie acumulada: caso 1, casos 1-2, casos 1-3...
    traza = core.traza_metricas(revisados, registro_inferencias)
    if not traza.empty:
        # Lista de partes anatomicas disponible para comparar el comportamiento global vs un subgrupo.
        opciones_parte = ["Todas"] + sorted(revisados["parte_anatomica"].dropna().astype(str).unique())

        # Dos columnas: izquierda = grafica global; derecha = grafica filtrada por parte anatomica.
        col_general, col_parte = st.columns(2)
        with col_general:
            # Este div solo crea un pequeno espacio vertical para alinear la grafica con la de la derecha, que tiene encima un selector.
            st.markdown("<div style='height: 86px'></div>", unsafe_allow_html=True)
            grafica_lineas_acotada(
                traza,
                ["recall", "fnr", "f1", "accuracy"],
                "Rendimiento clinico acumulado - general",
                "Valor",
                1.0,
            )
        with col_parte:
            # st.selectbox crea un desplegable. Al cambiarlo, Streamlit recarga la app y recalcula las graficas con la parte seleccionada.
            parte_elegida = st.selectbox(
                "Parte anatomica",
                opciones_parte,
                key="parte_anatomica_trace",
            )
            # Si se elige una parte concreta, filtramos los casos revisados antes de recalcular la traza.
            revisados_parte = revisados if parte_elegida == "Todas" else revisados[revisados["parte_anatomica"] == parte_elegida]
            referencia_parte = registro_inferencias if parte_elegida == "Todas" else registro_inferencias[registro_inferencias["parte_anatomica"] == parte_elegida]
            traza_parte = core.traza_metricas(revisados_parte, referencia_parte)
            titulo_parte = parte_elegida if parte_elegida != "Todas" else "todas las partes"
            grafica_lineas_acotada(
                traza_parte,
                ["recall", "fnr", "f1", "accuracy"],
                f"Rendimiento clinico acumulado - {titulo_parte}",
                "Valor",
                1.0,
            )

        # Segunda fila de graficas: no mide rendimiento puro, sino uso y seguridad.
        # Desacuerdo = discrepancia modelo/urgencias; error_alta_conf = fallo con el modelo muy seguro.
        col_uso_general, col_uso_parte = st.columns(2)
        with col_uso_general:
            grafica_lineas_acotada(
                traza,
                ["desacuerdo", "error_alta_conf_rate", "drift_global"],
                "Uso y seguridad acumulada - general",
                "Tasa",
                1.0,
            )
        with col_uso_parte:
            grafica_lineas_acotada(
                traza_parte,
                ["desacuerdo", "error_alta_conf_rate", "drift_global"],
                f"Uso y seguridad acumulada - {titulo_parte}",
                "Tasa",
                1.0,
            )

        # Para que la grafica de latencia tenga una escala razonable, calculamos el maximo observado y dejamos un pequeno margen superior multiplicando por 1.25.
        valores_latencia = pd.concat([
            traza["latencia_p95"].dropna(),
            traza["latencia_caso_ms"].dropna(),
        ])
        max_latencia = valores_latencia.max() if not valores_latencia.empty else 1.0
        limite_latencia = max(1.0, float(max_latencia) * 1.25) if pd.notna(max_latencia) else 1.0
        grafica_lineas_acotada(
            traza,
            ["latencia_caso_ms", "latencia_p95"],
            "Latencia del modelo",
            "Milisegundos",
            limite_latencia,
        )

        # st.expander es un desplegable plegable. Lo usamos para esconder una tabla larga que sirve para revisar los valores exactos de cada punto de las graficas.
        with st.expander("Ver serie acumulada caso a caso"):
            columnas_serie = [
                "caso_n", "id_estudio", "n", "n_normal", "n_anormal",
                "recall", "fnr", "f1", "accuracy",
                "acierto_normal", "acierto_anormal",
                "desacuerdo", "desacuerdo_normal", "desacuerdo_anormal",
                "error_alta_conf", "error_alta_conf_rate",
                "error_alta_conf_rate_normal", "error_alta_conf_rate_anormal",
                "drift_parte_anatomica", "drift_probabilidad", "drift_prediccion", "drift_global",
                "latencia_caso_ms", "latencia_p95",
            ]
            st.dataframe(traza[columnas_serie], hide_index=True, use_container_width=True)

    # Las matrices resumen aciertos y errores de forma mas directa que las metricas.
    st.subheader("Matrices de confusion")
    st.caption(
        "Resumen directo de aciertos y errores con los casos revisados. "
        "Filas = ground truth operativo urgencias; columnas = prediccion del modelo."
    )

    # Se muestran dos matrices en paralelo: una global y otra filtrable por parte anatomica.
    col_matriz_global, col_matriz_parte = st.columns(2)
    with col_matriz_global:
        # Espacio de alineacion con el selector de la columna derecha.
        st.markdown("<div style='height: 82px'></div>", unsafe_allow_html=True)
        st.markdown("**Matriz global**")
        mostrar_matriz_confusion(revisados)
    with col_matriz_parte:
        # Esta segunda matriz permite ver si un subgrupo anatomico concentra errores.
        opciones_parte = ["Todas"] + sorted(revisados["parte_anatomica"].dropna().astype(str).unique())
        parte_elegida = st.selectbox("Parte anatomica", opciones_parte)
        revisados_parte = revisados if parte_elegida == "Todas" else revisados[revisados["parte_anatomica"] == parte_elegida]
        st.markdown(f"**{parte_elegida if parte_elegida != 'Todas' else 'Todas las partes'}**")
        mostrar_matriz_confusion(revisados_parte)

    st.subheader("Tablas de detalle")
    st.caption(
        "Tablas con la informacion resumida en las graficas. Se mantienen al final "
        "para revisar valores exactos y para incluirlas en el informe de vigilancia."
    )

    # Tabla calculada por el backend agrupando las metricas por parte anatomica.
    st.markdown("**Indicadores por parte anatomica**")
    por_anatomia = core.metricas_por_parte_anatomica(revisados, registro_inferencias)
    if not por_anatomia.empty:
        st.dataframe(por_anatomia, hide_index=True, use_container_width=True)

    # La calibracion agrupa casos por rangos de probabilidad para ver si el modelo
    # se equivoca mas cuando dice estar muy seguro.
    st.markdown("**Calibracion simple**")
    st.markdown(
        "Esta tabla divide los casos revisados segun la probabilidad de anormalidad que dio el modelo, "
        "por ejemplo `0.0-0.2`, `0.2-0.4` o `0.8-1.0`. "
        "El rango `0.0-0.2` significa que el modelo asigno entre 0% y 20% de probabilidad de anormalidad; "
        "el rango `0.8-1.0` significa entre 80% y 100%."
    )
    st.caption(
        "La tasa de error sale de comparar `etiqueta_predicha` con el ground truth de urgencias en cada rango: "
        "`errores / casos del rango`. Por ejemplo, si en el rango 0.0-0.2 hay 3 casos y 1 esta mal, "
        "la tasa de error es 1/3 = 33.3%. Los errores de alta confianza son fallos donde el modelo estaba "
        f"muy seguro de su prediccion, con confianza >= {core.UMBRAL_ALTA_CONFIANZA:.2f}."
    )
    calibracion = core.tabla_calibracion(revisados)
    if not calibracion.empty:
        st.dataframe(tabla_calibracion_visible(calibracion), hide_index=True, use_container_width=True)

    # El informe se construye en el backend en formato Markdown y Streamlit lo ofrece como descarga.
    informe = core.construir_informe_vigilancia(registro_inferencias, revisados, alertas, auditoria)
    st.download_button(
        "Descargar informe de vigilancia",
        data=informe,
        file_name=f"informe_vigilancia_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# =============================================================================
# INTERFAZ: Pestaña 3 - Alertas
# =============================================================================

def mostrar_alertas(alertas):
    """Pantalla para investigar y cerrar alertas emergentes."""

    # Esta pestana muestra las alertas que calcula el backend a partir de las revisiones.
    st.header("Alertas emergentes")
    if alertas.empty:
        st.info("No hay alertas activas.")
        return

    # Combinamos las alertas generadas con los cambios de estado guardados por el usuario.
    # Si una alerta puede pasar de abierta a reconocida, en investigacion o cerrada.
    vista = core.combinar_estado_alertas(alertas)
    st.caption(
        "Las alertas son patrones emergentes calculados por el backend a partir de los casos revisados. "
        "No son diagnosticos clinicos; sirven para priorizar investigacion y documentar vigilancia. "
        "Por defecto se muestran solo las alertas no cerradas."
    )

    # Explicación plegable para que la pantalla no se llene de texto, pero tengas contexto si lo necesitas.
    with st.expander("Como interpretar las alertas"):
        st.markdown(
            "- **CLIN-03** aparece cuando urgencias marca una anormalidad y el modelo predice normal con mucha seguridad. "
            "En codigo equivale a `etiqueta_final=1`, `etiqueta_predicha=0` y confianza alta en normalidad.\n"
            "- **CAL-01** aparece cuando el modelo se equivoca con confianza alta. La confianza se calcula como "
            "`probabilidad_anormalidad` si predice anormal, y como `1 - probabilidad_anormalidad` si predice normal. "
            f"El umbral actual es `{core.UMBRAL_ALTA_CONFIANZA:.2f}`.\n"
            "- **DATA-01** aparece cuando cambia bastante la distribucion de los casos revisados respecto al registro base. "
            "Se compara la mezcla de partes anatomicas, la distribucion de probabilidades y la proporcion de predicciones normal/anormal.\n"
            "- **TECH-01** usa la latencia medida por el notebook durante la inferencia offline del modelo. "
            "No mide el tiempo que Streamlit tarda en leer el CSV."
        )

    # Filtros visuales. multiselect permite seleccionar varias severidades, selectbox permite elegir un unico estado.
    col_severidad, col_estado = st.columns(2)
    severidad = col_severidad.multiselect("Severidad", ["red", "orange", "yellow"], default=["red", "orange", "yellow"])
    estado = col_estado.selectbox(
        "Estado",
        ["Activas", "Todos", "open", "acknowledged", "investigating", "closed"],
        format_func=lambda valor: core.ETIQUETAS_ESTADO.get(valor, valor),
    )

    # Aplicamos los filtros sobre una copia/vista de las alertas.
    vista = vista[vista["severidad"].isin(severidad)]
    if estado == "Activas":
        vista = vista[vista["estado_efectivo"] != "closed"]
    elif estado != "Todos":
        vista = vista[vista["estado_efectivo"] == estado]

    # Resumen numerico superior de las alertas visibles tras aplicar filtros.
    conteo = vista["severidad"].value_counts().to_dict()
    valores = [len(vista), conteo.get("red", 0), conteo.get("orange", 0), conteo.get("yellow", 0)]
    for columna, etiqueta, valor in zip(st.columns(4), ["Total", "Rojas", "Naranjas", "Amarillas"], valores):
        _ = columna.metric(etiqueta, valor)

    if vista.empty:
        st.info("No hay alertas con estos filtros.")
        return

    # Ordenamos primero por severidad para que las alertas mas graves aparezcan arriba.
    vista["orden"] = vista["severidad"].map(core.ORDEN_SEVERIDAD).fillna(9)
    for _, alerta in vista.sort_values(["orden", "fecha_creacion"], ascending=[True, False]).iterrows():
        color_alerta = COLOR_SEVERIDAD.get(str(alerta["severidad"]), "#94a3b8")
        texto_estado = core.ETIQUETAS_ESTADO.get(alerta["estado_efectivo"], alerta["estado_efectivo"])

        # Streamlit no tiene una tarjeta de alerta tan especifica, asi que usamos HTML sencillo
        # dentro de st.markdown para pintar una barra lateral con el color de severidad.
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color_alerta}; padding: 0.35rem 0.75rem; margin: 0.35rem 0; background: rgba(148, 163, 184, 0.08);">
                <strong style="color:{color_alerta};">{str(alerta['severidad']).upper()}</strong>
                &nbsp; {alerta['codigo_alerta']} · {alerta['alcance']} · {texto_estado}
            </div>
            """,
            unsafe_allow_html=True,
        )
        titulo = f"{alerta['codigo_alerta']} | {alerta['alcance']} | {texto_estado}"

        # Cada alerta se abre en un expander para no saturar la pantalla si hay muchas.
        with st.expander(titulo, expanded=alerta["severidad"] == "red" and alerta["estado_efectivo"] == "open"):
            st.write(alerta["nombre_alerta"])
            st.markdown(f"**Patron:** {alerta.get('patron_emergente', '')}")
            st.markdown(f"**Alcance/subgrupo afectado:** {alerta.get('grupo_afectado', '')}")
            st.markdown(f"**Destinatario:** {core.ETIQUETAS_DESTINO.get(str(alerta.get('destinatario', '')), alerta.get('destinatario', ''))}")
            st.markdown(f"**Valor observado:** {alerta['valor_observado']} | **Umbral:** {alerta['umbral']} | **Muestra:** {alerta['tamano_muestra']}")
            st.markdown(f"**Accion recomendada:** {alerta['accion_recomendada']}")
            if str(alerta.get("evidencias", "")).strip():
                st.markdown(f"**Evidencia:** `{alerta['evidencias']}`")

            # st.form agrupa los campos y evita que cada cambio en el selectbox recargue la app.
            # Solo se procesa cuando se pulsa Guardar estado.
            with st.form(f"estado_{alerta['id_alerta']}"):
                nuevo_estado = st.selectbox("Nuevo estado", ["acknowledged", "investigating", "closed"], format_func=lambda valor: core.ETIQUETAS_ESTADO[valor])
                motivo_cierre = st.text_input("Motivo de cierre") if nuevo_estado == "closed" else ""
                accion_realizada = st.text_area("Accion realizada", height=70) if nuevo_estado == "closed" else ""
                requiere_seguimiento = st.selectbox("Requiere seguimiento", ["si", "no"]) if nuevo_estado == "closed" else ""
                guardar_estado = st.form_submit_button("Guardar estado")

            if guardar_estado:
                if nuevo_estado == "closed" and not motivo_cierre.strip():
                    st.warning("Indica un motivo para cerrar la alerta.")
                else:
                    # Cambiamos el estado en el CSV de alertas y registramos el cambio en auditoria.
                    core.cambiar_estado_alerta(str(alerta["id_alerta"]), nuevo_estado, motivo_cierre, accion_realizada, "responsable_vigilancia", requiere_seguimiento)
                    core.guardar_eventos_auditoria([core.evento_estado_alerta(alerta, nuevo_estado, motivo_cierre, accion_realizada, requiere_seguimiento)])
                    # st.rerun recarga la interfaz para que el nuevo estado se vea al momento.
                    st.rerun()


# =============================================================================
# INTERFAZ: Pestaña 4 - Auditoria
# =============================================================================

def mostrar_auditoria(auditoria):
    """Registro de trazabilidad de decisiones, alertas y cambios de estado."""

    # La auditoría es una tabla historica de decisiones, alertas generadas y cambios de estado.
    st.header("Auditoria")
    if auditoria.empty:
        st.info("Todavia no hay eventos.")
        return

    # Texto de contexto para recordar que no se sobreescriben eventos anteriores.
    st.caption(
        "La auditoria es un registro historico: no borra eventos antiguos. "
        "Por eso una alerta puede aparecer primero como generada y mas tarde como cerrada."
    )

    # Expander didactico para explicar columnas que pueden resultar ambiguas.
    with st.expander("Como leer actor y estado"):
        st.markdown(
            "- **Actor/origen** indica quien creo el evento: `Medico de urgencias` guarda una valoracion, "
            "`Motor de vigilancia` genera una alerta y `Responsable de vigilancia` cambia su estado.\n"
            "- **Estado en ese evento** es el estado registrado en esa fila historica. "
            "Si una alerta se cierra despues, las filas antiguas no se modifican; se anade un nuevo evento de cierre."
        )

    # Limpiamos NaN para que en la tabla no aparezcan valores raros, y ordenamos de mas reciente a mas antiguo.
    auditoria = auditoria.fillna("").sort_values("fecha", ascending=False)

    # Tres filtros en una fila:tipo de evento, estado registrado y busqueda libre.
    col_tipo, col_estado, col_busqueda = st.columns(3)
    tipo_evento = col_tipo.selectbox(
        "Tipo",
        ["Todos"] + sorted(auditoria["tipo_evento"].astype(str).unique()),
        format_func=lambda valor: core.ETIQUETAS_EVENTO.get(valor, valor),
    )
    valores_estado = sorted([valor for valor in auditoria["estado_alerta"].astype(str).unique() if valor])
    estado_registrado = col_estado.selectbox(
        "Estado registrado",
        ["Todos"] + valores_estado,
        format_func=lambda valor: core.ETIQUETAS_ESTADO.get(valor, valor),
    )
    busqueda = col_busqueda.text_input("Buscar caso o alerta")

    # Aplicamos los filtros. La tabla original no se modifica: solo creamos la vista que se muestra.
    vista = auditoria.copy()
    if tipo_evento != "Todos":
        vista = vista[vista["tipo_evento"] == tipo_evento]
    if estado_registrado != "Todos":
        vista = vista[vista["estado_alerta"] == estado_registrado]
    if busqueda.strip():
        texto_busqueda = busqueda.strip().lower()
        # Buscamos el texto en varias columnas utiles: caso, alerta o traza.
        vista = vista[
            vista["id_caso"].astype(str).str.lower().str.contains(texto_busqueda, na=False)
            | vista["id_alerta"].astype(str).str.lower().str.contains(texto_busqueda, na=False)
            | vista["id_traza"].astype(str).str.lower().str.contains(texto_busqueda, na=False)
        ]

    # Antes de pintar la tabla, convertimos codigos internos en etiquetas legibles.
    st.dataframe(tabla_auditoria_visible(vista), hide_index=True, use_container_width=True)

try:
    registro_inferencias, valoraciones, revisados, alertas, auditoria = core.cargar_estado()
except FileNotFoundError as exc:
    st.error(f"Falta un archivo necesario: {exc}")
    st.stop()

st.title("Sistema TFG de vigilancia poscomercializacion")
st.caption("Prototipo academico: el backend calcula metricas, calibracion, alertas emergentes, auditoria e informe.")

tab_revision, tab_vigilancia, tab_alertas, tab_auditoria = st.tabs(
    ["Revision clinica", "Vigilancia", "Alertas", "Auditoria"]
)

with tab_revision:
    mostrar_revision(registro_inferencias, valoraciones)
with tab_vigilancia:
    mostrar_vigilancia(registro_inferencias, revisados, alertas, auditoria)
with tab_alertas:
    mostrar_alertas(alertas)
with tab_auditoria:
    mostrar_auditoria(auditoria)
