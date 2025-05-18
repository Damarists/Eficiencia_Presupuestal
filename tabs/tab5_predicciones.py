import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def mostrar(modelos, datos_combinados):
    st.header("🔮 Predicciones Presupuestarias")
    st.markdown("""
    **¿Qué analiza?**  
    Predice ejecuciones presupuestarias utilizando Machine Learning para mejorar la planificación financiera.
    """)

    if st.button("Generar Predicciones", key="proy_btn"):
        with st.spinner('Generando predicciones...'):
            model, scaler = modelos['proyecciones_presupuestarias']
            X = datos_combinados[['PIA', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            X_scaled = scaler.transform(X)
            predicciones = model.predict(X_scaled)

            columnas_pred = [
                'PIM_Predicho',
                'Compromiso_Predicho',
                'Certificacion_Predicha',
                'Devengado_Predicho',
                'Girado_Predicho',
                'Avance_Porcentaje_Predicho'
            ]

            predicciones_df = pd.DataFrame(predicciones, columns=columnas_pred).reset_index(drop=True)
            metadatos = datos_combinados[['Tipo_Dataset', 'Nombre_Original', 'Año']].reset_index(drop=True)
            predicciones_df = pd.concat([metadatos, predicciones_df], axis=1)

            real_data = datos_combinados[[
                'PIM',
                'Compromiso Anual',
                'Certificación',
                'Ejecución_Devengado',
                'Ejecución_Girado',
                'Avance %'
            ]].reset_index(drop=True)

            real_data.columns = [
                'PIM_Real',
                'Compromiso_Real',
                'Certificacion_Real',
                'Devengado_Real',
                'Girado_Real',
                'Avance_Porcentaje_Real'
            ]

            resultados_completos = pd.concat([predicciones_df, real_data], axis=1)

            column_order = [
                'Tipo_Dataset',
                'Nombre_Original',
                'Año',
                'PIM_Predicho', 'PIM_Real',
                'Compromiso_Predicho', 'Compromiso_Real',
                'Certificacion_Predicha', 'Certificacion_Real',
                'Devengado_Predicho', 'Devengado_Real',
                'Girado_Predicho', 'Girado_Real',
                'Avance_Porcentaje_Predicho', 'Avance_Porcentaje_Real'
            ]

            st.subheader("📊 Comparativa: Predicciones vs Realidad")
            st.dataframe(
                resultados_completos[column_order].style.format({
                    'PIM_Predicho': 'S/ {:,.0f}',
                    'PIM_Real': 'S/ {:,.0f}',
                    'Compromiso_Predicho': 'S/ {:,.0f}',
                    'Compromiso_Real': 'S/ {:,.0f}',
                    'Certificacion_Predicha': 'S/ {:,.0f}',
                    'Certificacion_Real': 'S/ {:,.0f}',
                    'Devengado_Predicho': 'S/ {:,.0f}',
                    'Devengado_Real': 'S/ {:,.0f}',
                    'Girado_Predicho': 'S/ {:,.0f}',
                    'Girado_Real': 'S/ {:,.0f}',
                    'Avance_Porcentaje_Predicho': '{:,.2f}%',
                    'Avance_Porcentaje_Real': '{:,.2f}%'
                }).bar(
                    subset=['Avance_Porcentaje_Predicho', 'Avance_Porcentaje_Real'],
                    color='#5fba7d'
                ),
                height=500,
                use_container_width=True
            )

            st.success("✅ Predicciones generadas exitosamente con comparativa de datos reales")