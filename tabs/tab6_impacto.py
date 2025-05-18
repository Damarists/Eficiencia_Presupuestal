import streamlit as st
import pandas as pd
import numpy as np

def mostrar(modelos, datos_combinados):
    st.header("🏛 Impacto Organizacional (Cultura Organizacional)")
    st.markdown("""
    **¿Qué analiza?**  
    Mide el grado de aceptación institucional del uso de Machine Learning en la toma de decisiones presupuestarias.
    """)

    if st.button("Analizar Impacto", key="imp_btn"):
        with st.spinner('Analizando impacto organizacional...'):
            model = modelos['impacto_organizacional']
            X = datos_combinados[['Avance %', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            y_pred = model.predict(X)

            datos_combinados['Aceptacion_Predicha'] = y_pred

            st.subheader("Resultados de Aceptación Institucional")
            st.dataframe(datos_combinados[['Tipo_Dataset', 'Nombre_Original', 'Aceptacion_Predicha']])

            st.subheader("Notas sobre la Aceptación Institucional")
            st.info("""
             🔹 1: Avance de Ejecución Presupuestal mayor a 80%
             🔹 0.5: Avance de Ejecución Presupuestal entre 50% y 80%
             🔹 0: Avance de Ejecución Presupuestal menor a 50%
            """)

            st.success("✅ Impacto organizacional evaluado")