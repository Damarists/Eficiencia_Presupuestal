import pandas as pd
import joblib
import streamlit as st
from sklearn.calibration import LabelEncoder


@st.cache_resource
def cargar_modelos():
    modelos = joblib.load('modelos_presupuesto_unificados.pkl')
    modelos['modelo_multioutput'] = joblib.load('modelo_multioutput.pkl')
    modelos['modelo_ruboutput'] = joblib.load('modelo_ruboutput.pkl')
    return modelos


def procesar_datos(uploaded_file, tipo, label_encoder):
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

                    df['Nombre_encoded'] = df['Categoría_Presupuestal'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)
                    df['Nombre_Original'] = df['Categoría_Presupuestal']

                elif 'Productos/Proyectos' in df.columns:
                    nuevos_proyectos = set(df['Productos/Proyectos']) - set(label_encoder.classes_)
                    if nuevos_proyectos:
                        st.warning(
                            f"Se encontraron {len(nuevos_proyectos)} proyectos no vistos durante el entrenamiento")

                    df['Nombre_encoded'] = df['Productos/Proyectos'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)
                    df['Nombre_Original'] = df['Productos/Proyectos']

                elif 'Función' in df.columns:
                    nuevas_funciones = set(df['Función']) - set(label_encoder.classes_)
                    if nuevas_funciones:
                        st.warning(
                            f"Se encontraron {len(nuevas_funciones)} funciones no vistas durante el entrenamiento")

                    df['Nombre_encoded'] = df['Función'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)
                    df['Nombre_Original'] = df['Función']

                df['Tipo_Dataset_encoded'] = label_encoder.transform([tipo] * len(df))

            except Exception as e:
                st.error(f"Error al codificar categorías: {str(e)}")
                return None

            return df

        except Exception as e:
            st.error(f"Error crítico al procesar el archivo {tipo}: {str(e)}")
            return None
    return None


def procesar_financiamiento(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', thousands='.')
        df.columns = df.columns.str.strip()
        if 'Fuentes de Financimiento' in df.columns:
            df['Fuentes de Financimiento'] = df['Fuentes de Financimiento'].astype(str).str.extract(r'(\d+)').astype(
                int)
        return df
    return None