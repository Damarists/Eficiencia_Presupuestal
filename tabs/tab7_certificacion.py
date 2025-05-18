import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def mostrar(modelos, datos_combinados):
    st.header("📝 Certificación")
    st.markdown("""
    **¿Qué analiza?**  
    Detecta discrepancias en los procesos de certificación presupuestaria mediante Machine Learning.
    """)

    if st.button("Detectar Discrepancias en Certificación", key="cert_btn"):
        with st.spinner('Analizando discrepancias de certificación...'):
            model = modelos['certificacion']
            X = datos_combinados[['PIA', 'PIM', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            certificacion_predicha = model.predict(X)

            datos_combinados['Certificación_Predicha'] = certificacion_predicha
            datos_combinados['Discrepancia_Real'] = (
                datos_combinados['Certificación'] - datos_combinados['Certificación_Predicha']).abs()

            top_discrepancias = datos_combinados.sort_values('Discrepancia_Real', ascending=False).head(10)

            st.subheader("🔍 Top 10 Discrepancias en Certificación")
            columnas_mostrar = [
                'Tipo_Dataset',
                'Nombre_Original',
                'PIM',
                'Certificación',
                'Certificación_Predicha',
                'Discrepancia_Real'
            ]

            st.dataframe(
                top_discrepancias[columnas_mostrar].style.format({
                    'PIM': 'S/ {:,.0f}',
                    'Certificación': 'S/ {:,.0f}',
                    'Certificación_Predicha': 'S/ {:,.0f}',
                    'Discrepancia_Real': 'S/ {:,.0f}'
                }),
                height=400
            )

            st.subheader("📈 Comparación Certificación Real vs Predicha")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_discrepancias.set_index('Nombre_Original')[['Certificación', 'Certificación_Predicha']].plot(
                kind='bar',
                ax=ax,
                color=['#1f77b4', '#ff7f0e']
            )
            plt.title('Top 10 Discrepancias en Certificación')
            plt.ylabel('Monto (S/)')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)