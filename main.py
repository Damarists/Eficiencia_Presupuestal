from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

# Import all tab modules
from tabs import (
    tab1_optimizacion,
    tab2_monitoreo,
    tab3_decisiones,
    tab4_capacidad,
    tab5_predicciones,
    tab6_impacto,
    tab7_certificacion,
    tab8_compromiso,
    tab9_devengado,
    tab10_girado
)

st.set_page_config(page_title="Analizador Presupuestario", page_icon="💰", layout="wide")


@st.cache_resource
def cargar_modelos():
    modelos = joblib.load('modelos_presupuesto_unificados.pkl')
    modelos['modelo_multioutput'] = joblib.load('modelo_multioutput.pkl')
    modelos['modelo_ruboutput'] = joblib.load('modelo_ruboutput.pkl')
    return modelos


modelos = cargar_modelos()
label_encoder = modelos['label_encoder']

st.title("📊 Sistema Inteligente de Análisis Presupuestario")
st.markdown("""
Esta herramienta utiliza inteligencia artificial para analizar tus datos presupuestarios y proporcionar 
insights valiosos en 10 dimensiones clave. Sube tus archivos y obtén resultados comprensibles al instante.
""")

with st.sidebar:
    st.header("📤 Carga tus datos")
    st.markdown("Sube los archivos CSV para categoría, proyectos y función presupuestaria")

    uploaded_cat = st.file_uploader("Datos de Categoría Presupuestal", type=['csv'])
    uploaded_proy = st.file_uploader("Datos de Proyectos", type=['csv'])
    uploaded_func = st.file_uploader("Datos de Función", type=['csv'])
    uploaded_financiamiento = st.file_uploader("Fuentes de Financiamiento (por año)", type=['csv'])
    uploaded_rubro = st.file_uploader("Datos de Rubros (por año)", type=['csv'])

    st.markdown("---")
    st.markdown("🔍 **Cómo usar:**")
    st.markdown("1. Sube los tres archivos (formato CSV)")
    st.markdown("2. Espera a que se procesen los datos")
    st.markdown("3. Explora los resultados en las pestañas")
    st.markdown("4. Descarga los informes si lo necesitas")


def procesar_datos(uploaded_file, tipo):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';', decimal=',', thousands='.')

            money_cols = ['PIA', 'PIM', 'Certificación', 'Compromiso Anual',
                          'Ejecución_Atención_Compromiso_Mensual', 'Ejecución_Devengado', 'Ejecución_Girado']

            def clean_currency(value):
                if isinstance(value, str):
                    value = str(value).replace('S/', '').replace(' ', '').strip()
                    if value == '-' or value == '':
                        return 0.0
                    if '.' in value and ',' in value:
                        return float(value.replace('.', '').replace(',', '.'))
                    elif ',' in value:
                        return float(value.replace(',', '.'))
                    elif '.' in value:
                        parts = value.split('.')
                        if len(parts) > 1 and len(parts[-1]) == 2:
                            return float(value.replace('.', '', len(parts) - 1).replace('.', '.'))
                        else:
                            return float(value.replace('.', ''))
                return float(value) if value else 0.0

            for col in money_cols:
                if col in df.columns:
                    df[col] = df[col].apply(clean_currency)

            if 'Avance %' in df.columns:
                df['Avance %'] = df['Avance %'].astype(str)
                df['Avance %'] = df['Avance %'].str.replace('-', '0')
                df['Avance %'] = df['Avance %'].str.replace('%', '').str.replace(',', '.')
                df['Avance %'] = df['Avance %'].astype(float)

            df['Año'] = df['Año'].astype(int)
            df['Tipo_Dataset'] = tipo

            try:
                if 'Categoría_Presupuestal' in df.columns:
                    nuevas_categorias = set(df['Categoría_Presupuestal']) - set(label_encoder.classes_)
                    if nuevas_categorias:
                        st.warning(
                            f"Se encontraron {len(nuevas_categorias)} categorías no vistas durante el entrenamiento")
                        with st.expander("Ver categorías nuevas"):
                            st.write(list(nuevas_categorias)[:10])

                    df['Nombre_encoded'] = df['Categoría_Presupuestal'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
                    )
                    df['Nombre_Original'] = df['Categoría_Presupuestal']

                elif 'Productos/Proyectos' in df.columns:
                    nuevos_proyectos = set(df['Productos/Proyectos']) - set(label_encoder.classes_)
                    if nuevos_proyectos:
                        st.warning(
                            f"Se encontraron {len(nuevos_proyectos)} proyectos no vistos durante el entrenamiento")
                        with st.expander("Ver proyectos nuevos"):
                            st.write(list(nuevos_proyectos)[:10])

                    df['Nombre_encoded'] = df['Productos/Proyectos'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
                    )
                    df['Nombre_Original'] = df['Productos/Proyectos']

                elif 'Función' in df.columns:
                    nuevas_funciones = set(df['Función']) - set(label_encoder.classes_)
                    if nuevas_funciones:
                        st.warning(
                            f"Se encontraron {len(nuevas_funciones)} funciones no vistas durante el entrenamiento")
                        with st.expander("Ver funciones nuevas"):
                            st.write(list(nuevas_funciones)[:10])

                    df['Nombre_encoded'] = df['Función'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
                    )
                    df['Nombre_Original'] = df['Función']

                df['Tipo_Dataset_encoded'] = label_encoder.transform([tipo] * len(df))

            except Exception as e:
                st.error(f"Error al codificar categorías: {str(e)}")
                st.error("Categorías/proyectos/funciones problemáticos:")
                if 'Categoría_Presupuestal' in df.columns:
                    st.write(df['Categoría_Presupuestal'].unique()[:10])
                elif 'Productos/Proyectos' in df.columns:
                    st.write(df['Productos/Proyectos'].unique()[:10])
                elif 'Función' in df.columns:
                    st.write(df['Función'].unique()[:10])
                return None

            return df

        except Exception as e:
            st.error(f"Error crítico al procesar el archivo {tipo}: {str(e)}")
            if 'df' in locals():
                st.error("Primeras filas del archivo problemático:")
                st.dataframe(df.head())
            return None
    return None


def procesar_financiamiento(uploaded_file, label_encoder_fuente=None):
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
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', thousands='.')
        df.columns = df.columns.str.strip()

        # Procesar columna de Rubro
        if 'Rubro' in df.columns:
            df['Rubro'] = df['Rubro'].astype(str).str.extract(r'(\d+)').astype(int)

        # Procesar columnas monetarias
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

        # Procesar Avance %
        if 'Avance %' in df.columns:
            df['Avance %'] = df['Avance %'].astype(str) \
                .str.replace('%', '') \
                .str.replace(',', '.') \
                .str.strip()
            df['Avance %'] = pd.to_numeric(df['Avance %'], errors='coerce')

        return df
    return None

if uploaded_cat or uploaded_proy or uploaded_func:
    st.session_state.datos = {}

    if uploaded_cat:
        with st.spinner('Procesando datos de categoría...'):
            st.session_state.datos['categoria'] = procesar_datos(uploaded_cat, 'Categoría')

    if uploaded_proy:
        with st.spinner('Procesando datos de proyectos...'):
            st.session_state.datos['proyectos'] = procesar_datos(uploaded_proy, 'Proyecto')

    if uploaded_func:
        with st.spinner('Procesando datos de función...'):
            st.session_state.datos['funcion'] = procesar_datos(uploaded_func, 'Función')

    if st.session_state.datos:
        datos_combinados = pd.concat([df for df in st.session_state.datos.values() if df is not None])
        st.session_state.datos_combinados = datos_combinados
        st.success("¡Datos procesados correctamente!")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "🔍 Optimización Recursos",
            "📈 Análisis y Monitoreo",
            "🎯 Decisiones Estratégicas",
            "💻 Capacidad Tecnológica",
            "🔮 Predicciones",
            "🏛 Impacto Organizacional",
            "📝 Certificación",
            "🤝 Compromiso",
            "💰 Ejecución Devengado(Acumulado)",
            "💸 Girado"
        ])

        with tab1:
            tab1_optimizacion.mostrar(modelos, datos_combinados, uploaded_financiamiento, uploaded_rubro)

        with tab2:
            tab2_monitoreo.mostrar(modelos, datos_combinados)

        with tab3:
            tab3_decisiones.mostrar(modelos, datos_combinados)

        with tab4:
            tab4_capacidad.mostrar(modelos, datos_combinados)

        with tab5:
            tab5_predicciones.mostrar(modelos, datos_combinados)

        with tab6:
            tab6_impacto.mostrar(modelos, datos_combinados)

        with tab7:
            tab7_certificacion.mostrar(modelos, datos_combinados)

        with tab8:
            tab8_compromiso.mostrar(modelos, datos_combinados)

        with tab9:
            tab9_devengado.mostrar(modelos, datos_combinados)

        with tab10:
            tab10_girado.mostrar(modelos, datos_combinados)

else:
    st.info("👈 Por favor sube los archivos CSV en la barra lateral para comenzar el análisis")

# Footer
st.markdown("---")
st.markdown("© 2025 Sistema Inteligente de Análisis Presupuestario - Versión 1.0")