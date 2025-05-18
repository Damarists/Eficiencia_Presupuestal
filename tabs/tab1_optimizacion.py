import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

CATEGORIA_PRESUPUESTAL_DATA = {
    "RECURSOS ORDINARIOS": [
        {"Categoría_Presupuestal": "0001: PROGRAMA ARTICULADO NUTRICIONAL", "PIA": "S/ 0", "PIM": "S/ 159.143,00", "Certificación": "S/ 127.864,00", "Compromiso Anual": "S/ 127.864,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 127.864,00", "Ejecución_Devengado": "S/ 127.864,00", "Ejecución_Girado": "S/ 127.864,00", "Avance %": "80,3%", "Año": 2019},
        {"Categoría_Presupuestal": "0068: REDUCCION DE VULNERABILIDAD Y ATENCION DE EMERGENCIAS POR DESASTRES", "PIA": "S/ 0", "PIM": "S/ 100.000,00", "Certificación": "S/ 99.999,00", "Compromiso Anual": "S/ 99.999,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 99.999,00", "Ejecución_Devengado": "S/ 99.999,00", "Ejecución_Girado": "S/ 99.999,00", "Avance %": "100%", "Año": 2019},
        {"Categoría_Presupuestal": "9002: ASIGNACIONES PRESUPUESTARIAS QUE NO RESULTAN EN PRODUCTOS", "PIA": "S/ 336.632,00", "PIM": "S/ 336.632,00", "Certificación": "S/ 336.632,00", "Compromiso Anual": "S/ 336.632,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 336.632,00", "Ejecución_Devengado": "S/ 336.632,00", "Ejecución_Girado": "S/ 336.632,00", "Avance %": "100%", "Año": 2019},
    ],
    "RECURSOS DIRECTAMENTE RECAUDADOS": [
        {"Categoría_Presupuestal": "0001: PROGRAMA ARTICULADO NUTRICIONAL", "PIA": "S/ 0", "PIM": "S/ 15.000,00", "Certificación": "S/ 15.000,00", "Compromiso Anual": "S/ 15.000,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 15.000,00", "Ejecución_Devengado": "S/ 15.000,00", "Ejecución_Girado": "S/ 15.000,00", "Avance %": "100%", "Año": 2019},
        {"Categoría_Presupuestal": "0030: REDUCCION DE DELITOS Y FALTAS QUE AFECTAN LA SEGURIDAD CIUDADANA", "PIA": "S/ 101.123,00", "PIM": "S/ 172.242,00", "Certificación": "S/ 58.794,00", "Compromiso Anual": "S/ 58.794,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 58.794,00", "Ejecución_Devengado": "S/ 58.794,00", "Ejecución_Girado": "S/ 58.794,00", "Avance %": "34,1%", "Año": 2019},
        {"Categoría_Presupuestal": "0036: GESTION INTEGRAL DE RESIDUOS SOLIDOS", "PIA": "S/ 558.500,00", "PIM": "S/ 178.688,00", "Certificación": "S/ 146.144,00", "Compromiso Anual": "S/ 146.144,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 146.144,00", "Ejecución_Devengado": "S/ 146.144,00", "Ejecución_Girado": "S/ 146.144,00", "Avance %": "81,8%", "Año": 2019},
        {"Categoría_Presupuestal": "0041: MEJORA DE LA INOCUIDAD AGROALIMENTARIA", "PIA": "S/ 5.000,00", "PIM": "S/ 5.000,00", "Certificación": "S/ 0", "Compromiso Anual": "S/ 0", "Ejecución_Atención_Compromiso_Mensual": "S/ 0", "Ejecución_Devengado": "S/ 0", "Ejecución_Girado": "S/ 0", "Avance %": "0%", "Año": 2019},
        {"Categoría_Presupuestal": "0101: INCREMENTO DE LA PRACTICA DE ACTIVIDADES FISICAS, DEPORTIVAS Y RECREATIVAS EN LA POBLACION PERUANA", "PIA": "S/ 13.000,00", "PIM": "S/ 13.000,00", "Certificación": "S/ 0", "Compromiso Anual": "S/ 0", "Ejecución_Atención_Compromiso_Mensual": "S/ -", "Ejecución_Devengado": "S/ -", "Ejecución_Girado": "S/ 0", "Avance %": "0%", "Año": 2019},
        {"Categoría_Presupuestal": "0142: ACCESO DE PERSONAS ADULTAS MAYORES A SERVICIOS ESPECIALIZADOS", "PIA": "S/ 7.000,00", "PIM": "S/ 7.000,00", "Certificación": "S/ 0", "Compromiso Anual": "S/ 0", "Ejecución_Atención_Compromiso_Mensual": "S/ 0", "Ejecución_Devengado": "S/ 0", "Ejecución_Girado": "S/ 0", "Avance %": "0%", "Año": 2019},
        {"Categoría_Presupuestal": "9001: ACCIONES CENTRALES", "PIA": "S/ 624.901,00", "PIM": "S/ 981.793,00", "Certificación": "S/ 878.840,00", "Compromiso Anual": "S/ 826.755,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 858.399,00", "Ejecución_Devengado": "S/ 853.249,00", "Ejecución_Girado": "S/ 852.599,00", "Avance %": "86,9%", "Año": 2019},
        {"Categoría_Presupuestal": "9002: ASIGNACIONES PRESUPUESTARIAS QUE NO RESULTAN EN PRODUCTOS", "PIA": "S/ 71.000,00", "PIM": "S/ 386.763,00", "Certificación": "S/ 249.482,00", "Compromiso Anual": "S/ 249.248,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 249.248,00", "Ejecución_Devengado": "S/ 249.248,00", "Ejecución_Girado": "S/ 249.248,00", "Avance %": "64,4%", "Año": 2019},
    ],
    "RECURSOS POR OPERACIONES OFICIALES DE CREDITO": [
        {"Categoría_Presupuestal": "0082: PROGRAMA NACIONAL DE SANEAMIENTO URBANO", "PIA": "S/ 0", "PIM": "S/ 3.152.211,00", "Certificación": "S/ 3.151.406,00", "Compromiso Anual": "S/ 3.151.406,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 3.151.406,00", "Ejecución_Devengado": "S/ 3.151.406,00", "Ejecución_Girado": "S/ 3.151.406,00", "Avance %": "100%", "Año": 2019},
    ],
    "RECURSOS DETERMINADOS": [
        {"Categoría_Presupuestal": "0001: PROGRAMA ARTICULADO NUTRICIONAL", "PIA": "S/ 0", "PIM": "S/ 87.964,00", "Certificación": "S/ 87.964,00", "Compromiso Anual": "S/ 87.960,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 87.960,00", "Ejecución_Devengado": "S/ 87.960,00", "Ejecución_Girado": "S/ 87.960,00", "Avance %": "100%", "Año": 2019},
        {"Categoría_Presupuestal": "0030: REDUCCION DE DELITOS Y FALTAS QUE AFECTAN LA SEGURIDAD CIUDADANA", "PIA": "S/ 456.568,00", "PIM": "S/ 337.626,00", "Certificación": "S/ 325.012,00", "Compromiso Anual": "S/ 325.012,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 324.957,00", "Ejecución_Devengado": "S/ 324.957,00", "Ejecución_Girado": "S/ 324.957,00", "Avance %": "96,2%", "Año": 2019},
        {"Categoría_Presupuestal": "0036: GESTION INTEGRAL DE RESIDUOS SOLIDOS", "PIA": "S/ 1.830.453,00", "PIM": "S/ 2.572.062,00", "Certificación": "S/ 2.335.533,00", "Compromiso Anual": "S/ 2.315.869,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 2.315.131,00", "Ejecución_Devengado": "S/ 2.315.131,00", "Ejecución_Girado": "S/ 2.315.131,00", "Avance %": "90%", "Año": 2019},
        {"Categoría_Presupuestal": "0041: MEJORA DE LA INOCUIDAD AGROALIMENTARIA", "PIA": "S/ 10.000,00", "PIM": "S/ 0", "Certificación": "S/ 0", "Compromiso Anual": "S/ 0", "Ejecución_Atención_Compromiso_Mensual": "S/ 0", "Ejecución_Devengado": "S/ 0", "Ejecución_Girado": "S/ 0", "Avance %": "0%", "Año": 2019},
        {"Categoría_Presupuestal": "0046: ACCESO Y USO DE LA ELECTRIFICACION RURAL", "PIA": "S/ 0", "PIM": "S/ 42.473,00", "Certificación": "S/ 37.800,00", "Compromiso Anual": "S/ 37.800,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 37.800,00", "Ejecución_Devengado": "S/ 37.800,00", "Ejecución_Girado": "S/ 37.800,00", "Avance %": "89%", "Año": 2019},
        {"Categoría_Presupuestal": "0068: REDUCCION DE VULNERABILIDAD Y ATENCION DE EMERGENCIAS POR DESASTRES", "PIA": "S/ 38.000,00", "PIM": "S/ 27.057,00", "Certificación": "S/ 9.057,00", "Compromiso Anual": "S/ 9.057,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 9.057,00", "Ejecución_Devengado": "S/ 9.057,00", "Ejecución_Girado": "S/ 9.057,00", "Avance %": "33,5%", "Año": 2019},
        {"Categoría_Presupuestal": "0082: PROGRAMA NACIONAL DE SANEAMIENTO URBANO", "PIA": "S/ 1.145.002,00", "PIM": "S/ 2.962.488,00", "Certificación": "S/ 2.457.839,00", "Compromiso Anual": "S/ 841.873,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 841.873,00", "Ejecución_Devengado": "S/ 841.873,00", "Ejecución_Girado": "S/ 841.873,00", "Avance %": "28,4%", "Año": 2019},
        {"Categoría_Presupuestal": "0083: PROGRAMA NACIONAL DE SANEAMIENTO RURAL", "PIA": "S/ 1.715.000,00", "PIM": "S/ 1.436.529,00", "Certificación": "S/ 1.436.482,00", "Compromiso Anual": "S/ 844.187,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 844.187,00", "Ejecución_Devengado": "S/ 844.186,00", "Ejecución_Girado": "S/ 844.186,00", "Avance %": "58,8%", "Año": 2019},
        {"Categoría_Presupuestal": "0090: LOGROS DE APRENDIZAJE DE ESTUDIANTES DE LA EDUCACION BASICA REGULAR", "PIA": "S/ 0", "PIM": "S/ 1.109.061,00", "Certificación": "S/ 1.086.745,00", "Compromiso Anual": "S/ 827.544,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 827.544,00", "Ejecución_Devengado": "S/ 827.544,00", "Ejecución_Girado": "S/ 827.544,00", "Avance %": "74,6%", "Año": 2019},
        {"Categoría_Presupuestal": "0101: INCREMENTO DE LA PRACTICA DE ACTIVIDADES FISICAS, DEPORTIVAS Y RECREATIVAS EN LA POBLACION PERUANA", "PIA": "S/ 12.000,00", "PIM": "S/ 0", "Certificación": "S/ 0", "Compromiso Anual": "S/ 0", "Ejecución_Atención_Compromiso_Mensual": "S/ -", "Ejecución_Devengado": "S/ 0", "Ejecución_Girado": "S/ 0", "Avance %": "0%", "Año": 2019},
        {"Categoría_Presupuestal": "0109: NUESTRAS CIUDADES", "PIA": "S/ 0", "PIM": "S/ 333.963,00", "Certificación": "S/ 0", "Compromiso Anual": "S/ 0", "Ejecución_Atención_Compromiso_Mensual": "S/ 0", "Ejecución_Devengado": "S/ 0", "Ejecución_Girado": "S/ -", "Avance %": "0%", "Año": 2019},
        {"Categoría_Presupuestal": "0142: ACCESO DE PERSONAS ADULTAS MAYORES A SERVICIOS ESPECIALIZADOS", "PIA": "S/ 8.000,00", "PIM": "S/ 0", "Certificación": "S/ 0", "Compromiso Anual": "S/ 0", "Ejecución_Atención_Compromiso_Mensual": "S/ 0", "Ejecución_Devengado": "S/ 0", "Ejecución_Girado": "S/ 0", "Avance %": "0%", "Año": 2019},
        {"Categoría_Presupuestal": "0146: ACCESO DE LAS FAMILIAS A VIVIENDA Y ENTORNO URBANO ADECUADO", "PIA": "S/ 6.345.180,00", "PIM": "S/ 12.294.441,00", "Certificación": "S/ 9.343.269,00", "Compromiso Anual": "S/ 9.303.957,00", "Ejecución_Atención_Compromiso_Mensual": "S/ 9.303.957,00", "Ejecución_Devengado": "S/ 9.303.957,00", "Ejecución_Girado": "S/ 9.303.957,00", "Avance %": "74,2%", "Año": 2019}
  ]
}

RUBRO_TIPO_RECURSO_DATA = {
    "00: RECURSOS ORDINARIOS": [
        {"Tipo de Recurso": "B: SUB CUENTA - RECURSOS ORDINARIOS POR TRANSFERENCIA DE PARTIDAS", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 127864, "Ejecución_Devengado": 127864, "Ejecución_Girado": 127864, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "L: TRANSF. AL PROGRAMA DEL VASO DE LECHE - GL", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 336632, "Ejecución_Devengado": 336632, "Ejecución_Girado": 336632, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "0: RECURSOS ORDINARIOS", "PIA": 336632, "PIM": 595775, "Certificación": 564494, "Compromiso Anual": 564494, "Ejecución_Atención_Compromiso_Mensual": 0, "Ejecución_Devengado": 0, "Ejecución_Girado": 0, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "16: SUB CUENTA - ACTIVIDADES DE EMERGENCIA (R.O.)", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 99999, "Ejecución_Devengado": 99999, "Ejecución_Girado": 99999, "Avance %": 0, "Año": 2019},
    ],
    "07: FONDO DE COMPENSACION MUNICIPAL": [
        {"Tipo de Recurso": "A: SUB CUENTA - FONCOMUN", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 6050179, "Ejecución_Devengado": 6050003, "Ejecución_Girado": 6049997, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "0: NORMAL", "PIA": 5694612, "PIM": 7092705, "Certificación": 6304104, "Compromiso Anual": 6236011, "Ejecución_Atención_Compromiso_Mensual": 0, "Ejecución_Devengado": 0, "Ejecución_Girado": 0, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "2: FONCOMUN", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 108622, "Ejecución_Devengado": 108622, "Ejecución_Girado": 108622, "Avance %": 0, "Año": 2019},
    ],
    "08: IMPUESTOS MUNICIPALES": [
        {"Tipo de Recurso": "0: NORMAL", "PIA": 1284872, "PIM": 1586882, "Certificación": 1388189, "Compromiso Anual": 1314119, "Ejecución_Atención_Compromiso_Mensual": 1313360, "Ejecución_Devengado": 1313360, "Ejecución_Girado": 1313161, "Avance %": 0, "Año": 2019},
    ],
    "09: RECURSOS DIRECTAMENTE RECAUDADOS": [
        {"Tipo de Recurso": "0: NORMAL", "PIA": 1380524, "PIM": 1759486, "Certificación": 1348260, "Compromiso Anual": 1331940, "Ejecución_Atención_Compromiso_Mensual": 0, "Ejecución_Devengado": 0, "Ejecución_Girado": 0, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "1: UNIVERSIDADES/GOB.LOCALES", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 1327585, "Ejecución_Devengado": 1322435, "Ejecución_Girado": 1321785, "Avance %": 0, "Año": 2019},
    ],
    "18: CANON Y SOBRECANON, REGALIAS, RENTA DE ADUANAS Y PARTICIPACIONES": [
        {"Tipo de Recurso": "H: SUB CUENTA - CANON MINERO", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 7922193, "Ejecución_Devengado": 7922192, "Ejecución_Girado": 7922192, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "J: SUB CUENTA - CANON PESQUERO", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 62860, "Ejecución_Devengado": 62860, "Ejecución_Girado": 62860, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "O: SUB CUENTA - CANON PESQUERO, DERECHOS", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 80910, "Ejecución_Devengado": 80910, "Ejecución_Girado": 80910, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "P: SUB CUENTA - REGALIAS MINERAS", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 189740, "Ejecución_Devengado": 189740, "Ejecución_Girado": 189740, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "R: SUB CUENTA - FOCAM", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 70575, "Ejecución_Devengado": 69372, "Ejecución_Girado": 69372, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "0: CANON", "PIA": 12653098, "PIM": 21481940, "Certificación": 16618043, "Compromiso Anual": 8524980, "Ejecución_Atención_Compromiso_Mensual": 0, "Ejecución_Devengado": 0, "Ejecución_Girado": 0, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "13: SUBCUENTA- PLAN DE INCENTIVOS A LA MEJORA DE LA GESTION Y MODERNIZACION MUNICIPAL", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 125575, "Ejecución_Devengado": 125575, "Ejecución_Girado": 125518, "Avance %": 0, "Año": 2019},
    ],
    "19: RECURSOS POR OPERACIONES OFICIALES DE CREDITO": [
        {"Tipo de Recurso": "F: SUB CUENTA - ENDEUDAMIENTO - BONOS", "PIA": 0, "PIM": 0, "Certificación": 0, "Compromiso Anual": 0, "Ejecución_Atención_Compromiso_Mensual": 3151406, "Ejecución_Devengado": 3151406, "Ejecución_Girado": 3151406, "Avance %": 0, "Año": 2019},
        {"Tipo de Recurso": "0: ENDEUDAMIENTO EXTERNO", "PIA": 0, "PIM": 3152211, "Certificación": 3151406, "Compromiso Anual": 3151406, "Ejecución_Atención_Compromiso_Mensual": 0, "Ejecución_Devengado": 0, "Ejecución_Girado": 0, "Avance %": 0, "Año": 2019},
    ],
}

def mostrar(modelos, datos_combinados, uploaded_financiamiento, uploaded_rubro):
    st.header("🔍 Optimización de Recursos")
    st.markdown("""
    **¿Qué analiza?**  
    Evalúa cómo se están utilizando los recursos presupuestarios y identifica oportunidades para mejorar su asignación.
    """)
    if st.button("Analizar Optimización", key="opt_btn"):
        with st.spinner('Calculando optimización...'):
            X = datos_combinados[['PIA', 'PIM', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            y_pred = modelos['optimizacion_recursos'].predict(X)

            datos_combinados['Avance_Predicho'] = y_pred
            datos_combinados['Diferencia'] = datos_combinados['Avance %'] - datos_combinados['Avance_Predicho']

            st.session_state['opt_datos'] = datos_combinados.copy()
            st.session_state['opt_analizado'] = True

    if st.session_state.get('opt_analizado', False):
        datos_combinados = st.session_state['opt_datos']
        st.subheader("Resultados por la Ejecución Promedio")

        st.subheader("📌 Saldo Balance por Año")
        años_referencia = {
            2019: {'Saldo Balance': 5616219},
            2020: {'Saldo Balance': 12469494},
            2021: {'Saldo Balance': 14009719},
            2022: {'Saldo Balance': 3516116},
            2023: {'Saldo Balance': 23431732},
            2024: {'Saldo Balance': 25516531}
        }

        # Filtrar años existentes en los datos
        años_presentes = datos_combinados['Año'].unique()

        # Crear columnas para mostrar los valores
        cols = st.columns(3)
        col_idx = 0

        for año in sorted(años_presentes):
            if año in años_referencia:
                with cols[col_idx % 3]:
                    st.markdown(f"""
                                **Año {año}**  
                                Saldo Balance: S/ {años_referencia[año]['Saldo Balance']:,} 
                                """)
                col_idx += 1

        # --- Nuevos botones de filtrado ---

        # Inicializa el filtro activo si no existe
        if 'tab1_filtro_activo' not in st.session_state:
            st.session_state['tab1_filtro_activo'] = None
        st.subheader("Filtrar por tipo de dataset")
        col_cat, col_proy, col_func, col_financ, col_rubro = st.columns(5)
        if col_cat.button("Categoría", key="btn_cat"):
            st.session_state['tab1_filtro_activo'] = 'cat'
        if col_proy.button("Proyectos", key="btn_proy"):
            st.session_state['tab1_filtro_activo'] = 'proy'
        if col_func.button("Función", key="btn_func"):
            st.session_state['tab1_filtro_activo'] = 'func'
        if col_financ.button("Fuente", key="btn_financ"):
            st.session_state['tab1_filtro_activo'] = 'financ'
        if col_rubro.button("Rubro", key="btn_rubro"):
            st.session_state['tab1_filtro_activo'] = 'rubro'

        filtro_activo = st.session_state['tab1_filtro_activo']
        # --- Mostrar tabla de Rubros ---
        if filtro_activo == 'rubro':
            if uploaded_rubro is not None:
                df_rub = procesar_rubro(uploaded_rubro)
                df_rub.columns = df_rub.columns.str.strip()

                FINANCIAMIENTO_A_RUBRO = {
                    'RECURSOS ORDINARIOS': [0],
                    'RECURSOS DIRECTAMENTE RECAUDADOS': [9],
                    'RECURSOS POR OPERACIONES OFICIALES DE CREDITO': [19],
                    'RECURSOS DETERMINADOS': [7, 8, 18]
                }

                fuentes = st.session_state.get('selected_financiamiento', [])
                if fuentes:
                    rubros_a_mostrar = []
                    for fuente in fuentes:
                        rubros_a_mostrar.extend(FINANCIAMIENTO_A_RUBRO.get(fuente, []))
                    df_rub = df_rub[df_rub['Rubro'].isin(rubros_a_mostrar)]
                else:
                    # Mostrar todos los rubros del año por defecto (ejemplo: 2019)
                    df_rub = df_rub[df_rub['Año'] == 2019]

                rubro_map_anio = {
                    2019: {
                        0: "00: RECURSOS ORDINARIOS",
                        7: "07: FONDO DE COMPENSACION MUNICIPAL",
                        8: "08: IMPUESTOS MUNICIPALES",
                        9: "09: RECURSOS DIRECTAMENTE RECAUDADOS",
                        18: "18: CANON Y SOBRECANON, REGALIAS, RENTA DE ADUANAS Y PARTICIPACIONES",
                        19: "19: RECURSOS POR OPERACIONES OFICIALES DE CREDITO"
                    },
                    2020: {
                        0: "00: RECURSOS ORDINARIOS",
                        7: "07: FONDO DE COMPENSACION MUNICIPAL",
                        8: "08: IMPUESTOS MUNICIPALES",
                        9: "09: RECURSOS DIRECTAMENTE RECAUDADOS",
                        18: "18: CANON Y SOBRECANON, REGALIAS, RENTA DE ADUANAS Y PARTICIPACIONES",
                        19: "19: RECURSOS POR OPERACIONES OFICIALES DE CREDITO"
                    }
                }

                monetary_cols = ['PIA', 'PIM', 'Certificación', 'Compromiso Anual',
                                 'Ejecución_Atención_Compromiso_Mensual', 'Ejecución_Devengado',
                                 'Ejecución_Girado']
                for col in monetary_cols:
                    if col in df_rub.columns:
                        df_rub[col] = df_rub[col].astype(str).str.replace('S/', '', regex=False) \
                            .str.replace('.', '', regex=False) \
                            .str.replace(',', '.', regex=False) \
                            .str.strip() \
                            .replace('-', '0') \
                            .astype(float)
                if 'Avance %' in df_rub.columns:
                    df_rub['Avance %'] = df_rub['Avance %'].astype(str)
                    df_rub['Avance %'] = (
                        df_rub['Avance %']
                        .str.replace('%', '')
                        .str.replace(',', '.')
                        .str.strip()
                    )
                    df_rub['Avance %'] = pd.to_numeric(df_rub['Avance %'], errors='coerce')
                    df_rub = df_rub.dropna(subset=['Avance %'])

                    # Calcular promedio manualmente
                    suma_avances = df_rub['Avance %'].sum()
                    cantidad_avances = len(df_rub['Avance %'])
                    avance_promedio_real = suma_avances / cantidad_avances if cantidad_avances > 0 else 0
                    avance_esperado = 100  # Meta del 100%

                    st.subheader("📊 Resumen de Ejecución - Rubros")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Ejecución Promedio Real", f"{avance_promedio_real:.1f}%")
                    with col2:
                        st.metric("Ejecución Esperado", f"{avance_esperado}%",
                                  delta=f"{avance_promedio_real - avance_esperado:.1f}%")

                df_rub['Rubro_Nombre'] = [
                    rubro_map_anio.get(int(row['Año']), {}).get(int(row['Rubro']), "Otro")
                    for _, row in df_rub.iterrows()
                ]
                df_rub['Rubro_encoded'] = df_rub['Rubro'].astype(int)

                modelo_rub = modelos['modelo_ruboutput']
                X_rub = df_rub[['Año', 'Rubro_encoded']].rename(columns={'Rubro_encoded': 'Rubro'})
                pred_rub = modelo_rub.predict(X_rub)
                columnas_salida = ['PIA', 'PIM', 'Certificación', 'Compromiso Anual',
                                   'Ejecución_Atención_Compromiso_Mensual', 'Ejecución_Devengado',
                                   'Ejecución_Girado', 'Avance %']
                df_pred_rub = pd.DataFrame(pred_rub, columns=[col + ' Predicho' for col in columnas_salida])

                df_resultado = pd.concat([
                    df_rub[['Año', 'Rubro_Nombre']].reset_index(drop=True),
                    df_rub[columnas_salida].reset_index(drop=True),
                    df_pred_rub
                ], axis=1)

                columnas_finales = ['Año', 'Rubro_Nombre']
                for col in columnas_salida:
                    if col in df_rub.columns:
                        columnas_finales.append(col)
                        columnas_finales.append(col + ' Predicho')

                st.dataframe(df_resultado[columnas_finales])

                rubros_lista = list(RUBRO_TIPO_RECURSO_DATA.keys())
                if 'selected_rubro' not in st.session_state:
                    st.session_state.selected_rubro = []
                seleccion = st.multiselect("Selecciona un rubro", rubros_lista, default=st.session_state.selected_rubro)
                st.session_state['temp_selected_rubro'] = seleccion

                if st.button("Confirmar selección de rubro"):
                    st.session_state.selected_rubro = st.session_state['temp_selected_rubro']
                    st.success("Selección de rubro confirmada. Ahora puedes ver el tipo de recurso.")

                if st.session_state.selected_rubro:
                    rubro = st.session_state.selected_rubro[0]
                    if st.button(f"Ver tipo de recurso para: {rubro}"):
                        df_tipo = pd.DataFrame(RUBRO_TIPO_RECURSO_DATA[rubro])
                        ejecucion_promedio = df_tipo['Avance %'].sum()
                        ejecucion_esperado = 100
                        st.subheader(f"Detalle por tipo de recurso: {rubro}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Ejecución Promedio", f"{ejecucion_promedio:.1f}%")
                        with col2:
                            st.metric("Ejecución Esperado", f"{ejecucion_esperado}%")
                        st.dataframe(df_tipo)

                # Mostrar gráfico de comparación
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.bar(['Real', 'Esperado'], [avance_promedio_real, avance_esperado],
                       color=['#1f77b4', '#2ca02c'])
                ax.set_ylabel('Porcentaje (%)')
                ax.set_title('Comparación Ejecución Real vs Esperado')
                st.pyplot(fig)
                st.success("✅ Predicción de rubros generada")
            else:
                st.warning("Por favor sube el archivo de rubros para ver la predicción.")

        # --- Mostrar tabla de Financiamiento ---
        elif filtro_activo == 'financ':
            st.subheader("🔎 Fuentes de Financiamiento")
            if uploaded_financiamiento is not None:
                df_fin = procesar_financiamiento(uploaded_financiamiento)
                if 'modelo_multioutput' in modelos:
                    modelo_multi = modelos['modelo_multioutput']
                    X_multi = df_fin[['Año', 'Fuentes de Financimiento']].fillna(0)
                    pred_multi = modelo_multi.predict(X_multi)
                    columnas_salida = ['Avance %', 'PIA', 'PIM', 'Certificación', 'Compromiso Anual',
                                       'Ejecución_Atención_Compromiso_Mensual', 'Ejecución_Devengado',
                                       'Ejecución_Girado']
                    df_pred_multi = pd.DataFrame(pred_multi,
                                                 columns=[col + ' Predicho' for col in columnas_salida])

                    fuente_map_anio = {
                        2019: {
                            1: "RECURSOS ORDINARIOS",
                            2: "RECURSOS DIRECTAMENTE RECAUDADOS",
                            3: "RECURSOS POR OPERACIONES OFICIALES DE CREDITO",
                            5: "RECURSOS DETERMINADOS"
                        },
                        2020: {
                            1: "RECURSOS ORDINARIOS",
                            2: "RECURSOS DIRECTAMENTE RECAUDADOS",
                            3: "RECURSOS POR OPERACIONES OFICIALES DE CREDITO",
                            5: "RECURSOS DETERMINADOS"
                        },
                        2021: {
                            1: "RECURSOS ORDINARIOS",
                            2: "RECURSOS DIRECTAMENTE RECAUDADOS",
                            3: "RECURSOS POR OPERACIONES OFICIALES DE CREDITO",
                            5: "RECURSOS DETERMINADOS"
                        },
                        2022: {
                            1: "RECURSOS ORDINARIOS",
                            2: "RECURSOS DIRECTAMENTE RECAUDADOS",
                            3: "RECURSOS POR OPERACIONES OFICIALES DE CREDITO",
                            4: "DONACIONES Y TRANSFERENCIAS",
                            5: "RECURSOS DETERMINADOS"
                        },
                        2023: {
                            1: "RECURSOS ORDINARIOS",
                            2: "RECURSOS DIRECTAMENTE RECAUDADOS",
                            4: "DONACIONES Y TRANSFERENCIAS",
                            5: "RECURSOS DETERMINADOS"
                        },
                        2025: {
                            1: "RECURSOS ORDINARIOS",
                            2: "RECURSOS DIRECTAMENTE RECAUDADOS",
                            4: "DONACIONES Y TRANSFERENCIAS",
                            5: "RECURSOS DETERMINADOS"
                        }
                    }

                    df_fin['Fuente_Nombre'] = [
                        fuente_map_anio.get(int(row['Año']), {}).get(int(row['Fuentes de Financimiento']),
                                                                     "Otro")
                        for _, row in df_fin.iterrows()
                    ]

                    df_resultado = pd.concat([
                        df_fin[['Año', 'Fuente_Nombre']].reset_index(drop=True),
                        df_fin[columnas_salida].reset_index(drop=True),
                        df_pred_multi
                    ], axis=1)

                    if 'Avance %' in df_fin.columns:
                        # Convertir a string por si acaso hay valores no string
                        df_fin['Avance %'] = df_fin['Avance %'].astype(str)

                        # Eliminar símbolos % y espacios, reemplazar comas por puntos
                        df_fin['Avance %'] = (
                            df_fin['Avance %']
                            .str.replace('%', '')
                            .str.replace(',', '.')
                            .str.strip()
                        )

                        # Convertir a numérico, manejar valores inválidos
                        df_fin['Avance %'] = pd.to_numeric(df_fin['Avance %'], errors='coerce')

                        # Eliminar filas con valores NaN (si las hay)
                        df_fin = df_fin.dropna(subset=['Avance %'])

                        # Calcular promedio manualmente
                        suma_avances = df_fin['Avance %'].sum()
                        cantidad_avances = len(df_fin['Avance %'])
                        avance_promedio_real = suma_avances / cantidad_avances if cantidad_avances > 0 else 0

                        avance_esperado = 100  # Meta del 100%

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Ejecución Promedio Real", f"{avance_promedio_real:.1f}%")
                        with col2:
                            st.metric("Ejecución Esperado", f"{avance_esperado}%",
                                      delta=f"{avance_promedio_real - avance_esperado:.1f}%")

                    if 'selected_financiamiento' not in st.session_state:
                        st.session_state.selected_financiamiento = []

                    df_resultado['Seleccionar'] = df_resultado['Fuente_Nombre'].isin(
                        st.session_state.selected_financiamiento)

                    # Mostrar editor con checkboxes y guardar selección temporal
                    edited_df = st.data_editor(
                        df_resultado,
                        column_config={
                            "Seleccionar": st.column_config.CheckboxColumn(
                                help="Selecciona fuentes para filtrar rubros",
                                default=False
                            )
                        },
                        disabled=df_resultado.columns.tolist()[:-1]
                    )

                    # Selección temporal
                    nuevos_seleccionados = edited_df[edited_df['Seleccionar']]['Fuente_Nombre'].unique().tolist()
                    st.session_state['temp_selected_financiamiento'] = nuevos_seleccionados

                    # Botón para confirmar selección
                    if st.button("Confirmar selección de financiamiento"):
                        st.session_state.selected_financiamiento = st.session_state['temp_selected_financiamiento']
                        st.success("Selección confirmada. Ahora puedes presionar 'Rubro'.")

                    st.info("Los Recursos Ordinarios se debe gastar al 100% en el ejercicio presupuestal, en este caso el recurso ordinario solo se ha ejecutado al 94.8%, por lo que el restante se revierte al Estado")

                    # Mostrar gráfico de comparación
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.bar(['Real', 'Esperado'], [avance_promedio_real, avance_esperado],
                           color=['#1f77b4', '#2ca02c'])
                    ax.set_ylabel('Porcentaje (%)')
                    ax.set_title('Comparación Ejecución Real vs Esperado')
                    st.pyplot(fig)
                    st.success("✅ Predicción de fuentes de financiamiento generada")
                else:
                    st.info("No se encontró el modelo de fuentes de financiamiento en los modelos cargados.")
            else:
                st.warning("Por favor sube el archivo de fuentes de financiamiento para ver la predicción.")

        # --- Mostrar solo Categorías ---
        elif filtro_activo == 'cat':
            df_cat = datos_combinados[datos_combinados['Tipo_Dataset'] == 'Categoría']

            st.subheader("📊 Datos de Categoría Presupuestal")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Ejecución Real Promedio", f"{df_cat['Avance %'].mean():.1f}%")
            with cols[1]:
                st.metric("Ejecución Esperado", f"{100}%")

            st.dataframe(df_cat[['Nombre_Original', 'PIA', 'PIM', 'Avance %',
                                 'Avance_Predicho', 'Diferencia']].sort_values('Diferencia'))

            # Mostrar gráfico de comparación
            avance_promedio_real = df_cat['Avance %'].mean()  # Calcula el avance real promedio
            avance_esperado = 100  # Avance esperado (fijo en 100%)

            fuentes = st.session_state.get('selected_financiamiento', [])
            fuente = fuentes[0] if fuentes else None
            fuente_map = {
                "RECURSOS ORDINARIOS": "RECURSOS ORDINARIOS",
                "RECURSOS DIRECTAMENTE RECAUDADOS": "RECURSOS DIRECTAMENTE RECAUDADOS",
                "RECURSOS POR OPERACIONES OFICIALES DE CREDITO": "RECURSOS POR OPERACIONES OFICIALES DE CREDITO",
                "RECURSOS DETERMINADOS": "RECURSOS DETERMINADOS"
            }
            clave = fuente_map.get(fuente)
            if clave and clave in CATEGORIA_PRESUPUESTAL_DATA:
                df_cat = pd.DataFrame(CATEGORIA_PRESUPUESTAL_DATA[clave])
                df_cat['Avance %'] = df_cat['Avance %'].astype(str).str.replace('%', '').str.replace(',',
                                                                                                     '.').str.strip()
                df_cat['Avance %'] = pd.to_numeric(df_cat['Avance %'], errors='coerce').fillna(0)
                promedio_ejecucion = df_cat['Avance %'].mean()
                st.subheader(f"Detalle por Categoría Presupuestal - {fuente}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ejecución Promedio", f"{promedio_ejecucion:.1f}%")
                with col2:
                    st.metric("Ejecución Esperado", "100%")
                st.dataframe(df_cat)
            else:
                st.info("Selecciona una fuente de financiamiento para ver las categorías.")

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.bar(['Real', 'Esperado'], [avance_promedio_real, avance_esperado],
                   color=['#1f77b4', '#2ca02c'])
            ax.set_ylabel('Porcentaje (%)')
            ax.set_title('Comparación Ejecución Real vs Esperado')

            st.pyplot(fig)

        # --- Mostrar solo Proyectos ---
        elif filtro_activo == 'proy':
            df_proy = datos_combinados[datos_combinados['Tipo_Dataset'] == 'Proyecto']

            st.subheader("📊 Datos de Proyectos")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Ejecución Real Promedio", f"{df_proy['Avance %'].mean():.1f}%")
            with cols[1]:
                st.metric("Ejecución Esperado", f"{100}%")
            st.dataframe(df_proy[['Nombre_Original', 'PIA', 'PIM', 'Avance %',
                                  'Avance_Predicho', 'Diferencia']].sort_values('Diferencia'))

            # Mostrar gráfico de comparación
            avance_promedio_real = df_proy['Avance %'].mean()  # Calcula el avance real promedio
            avance_esperado = 100  # Avance esperado (fijo en 100%)

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.bar(['Real', 'Esperado'], [avance_promedio_real, avance_esperado],
                   color=['#1f77b4', '#2ca02c'])
            ax.set_ylabel('Porcentaje (%)')
            ax.set_title('Comparación Ejecución Real vs Esperado')

            st.pyplot(fig)

        # --- Mostrar solo Funciones ---
        elif filtro_activo == 'func':
            df_func = datos_combinados[datos_combinados['Tipo_Dataset'] == 'Función']

            st.subheader("📊 Datos de Función")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Ejecución Real Promedio", f"{df_func['Avance %'].mean():.1f}%")
            with cols[1]:
                st.metric("Ejecución Esperado", f"{100}%")
            st.dataframe(df_func[['Nombre_Original', 'PIA', 'PIM', 'Avance %',
                                  'Avance_Predicho', 'Diferencia']].sort_values('Diferencia'))

            # Mostrar gráfico de comparación
            avance_promedio_real = df_func['Avance %'].mean()  # Calcula el avance real promedio
            avance_esperado = 100  # Avance esperado (fijo en 100%)

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.bar(['Real', 'Esperado'], [avance_promedio_real, avance_esperado],
                   color=['#1f77b4', '#2ca02c'])
            ax.set_ylabel('Porcentaje (%)')
            ax.set_title('Comparación Ejecución Real vs Esperado')

            st.pyplot(fig)

        # --- Mostrar todos los datos si no se ha seleccionado ningún filtro ---
        else:
            st.subheader("Avance de Ejecución Promedio por Tipo de Dataset")
            cols = st.columns(3)
            statuses = []
            for i, (tipo, df) in enumerate(datos_combinados.groupby('Tipo_Dataset')):
                with cols[i]:
                    avance_real = df['Avance %'].mean()
                    avance_esperado = df['Avance_Predicho'].mean()

                    if avance_real > 90:
                        estado = "BUENO ✅"
                        descripcion = "Supera el 80% esperado"
                    elif avance_real > 80 and avance_esperado <= 80:
                        estado = "CASI BUENO 🟡"
                        descripcion = "Supera lo esperado pero no el 80%"
                    else:
                        estado = "MEJORABLE ⚠️"
                        descripcion = "No alcanza lo esperado"

                    statuses.append(estado)

                    st.metric(label=f"{tipo} - Avance de Ejecución Real",
                              value=f"{avance_real:.1f}%")
                    st.metric(label=f"{tipo} - Avance de Ejecución Esperado",
                              value=f"{100}%",
                              delta=descripcion)

                    st.metric(label="Eficiencia General", value=estado)

            with st.expander("📊 Ver detalles por item"):
                st.dataframe(datos_combinados[['Tipo_Dataset', 'Nombre_Original', 'Avance %',
                                               'Avance_Predicho', 'Diferencia']].sort_values('Diferencia'))

                count_buenos = statuses.count("BUENO ✅")
                count_casi_buenos = statuses.count("CASI BUENO 🟡")
                count_mejorables = statuses.count("MEJORABLE ⚠️")

                st.subheader("Recomendaciones")
                if count_buenos >= 2:
                    st.success("""
                    **Buen desempeño:**  
                    La asignación de recursos está siendo eficiente en general.  
                    🔹 Mantener los procesos actuales  
                    🔹 Monitorear áreas con pequeñas diferencias  
                    🔹 Replicar buenas prácticas en otras áreas
                    """)
                elif (count_casi_buenos + count_mejorables) >= 2:
                    st.warning("""
                    **Oportunidad de mejora:**  
                    La ejecución real está por debajo de lo esperado en varios rubros.  
                    🔹 Revisar los proyectos con mayor diferencia negativa  
                    🔹 Evaluar posibles cuellos de botella en la ejecución  
                    🔹 Considerar redistribución de recursos a áreas más eficientes
                    """)
                else:
                    st.info("""
                    **Desempeño mixto**  
                    Algunos rubros cumplen objetivos mientras otros requieren atención.  
                    🔹 Identificar mejores prácticas para replicar  
                    🔹 Implementar ajustes selectivos  
                    🔹 Monitorear indicadores clave
                    """)


def procesar_financiamiento(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', thousands='.')
        df.columns = df.columns.str.strip()
        if 'Fuentes de Financimiento' in df.columns:
            df['Fuentes de Financimiento'] = df['Fuentes de Financimiento'].astype(str).str.extract(r'(\d+)').astype(
                int)
        return df
    return None


def procesar_rubro(uploaded_file):
    if uploaded_file is not None:
        try:
            # Mantener separador original que funcionaba
            df = pd.read_csv(uploaded_file, sep=';', decimal=',', thousands='.')

            # Procesar columna Rubro (ej: "00: RECURSOS..." -> 0)
            df['Rubro'] = df['Rubro'].str.extract(r'(\d+)').astype(int)

            # Resto del procesamiento original
            monetary_cols = ['PIA', 'PIM', 'Certificación', 'Compromiso Anual',
                             'Ejecución_Atención_Compromiso_Mensual', 'Ejecución_Devengado',
                             'Ejecución_Girado']

            for col in monetary_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('S/', '', regex=False) \
                        .str.replace('.', '', regex=False) \
                        .str.replace(',', '.', regex=False) \
                        .str.strip() \
                        .replace('-', '0') \
                        .astype(float)

            if 'Avance %' in df.columns:
                df['Avance %'] = df['Avance %'].str.replace('%', '') \
                    .str.replace(',', '.') \
                    .astype(float)

            return df

        except Exception as e:
            st.error(f"Error procesando rubros: Verifica el formato del archivo. Detalle: {str(e)}")
            return None
    return None