import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def mostrar(modelos, datos_combinados):
    st.header("📈 Análisis y Monitoreo")
    st.markdown("""
    **¿Qué analiza?**  
    Proporciona una visión en tiempo real del desempeño presupuestario y alertas sobre posibles desviaciones.
    """)

    if st.button("Realizar Análisis", key="mon_btn"):
        with st.spinner('Analizando desempeño...'):
            model, scaler = modelos['analisis_monitoreo']
            X = datos_combinados[['PIA', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            X = X.fillna(0)
            X_scaled = scaler.transform(X)
            pim_predicho = model.predict(X_scaled)

            datos_combinados['PIM_Predicho'] = pim_predicho
            datos_combinados['Error_Soles'] = datos_combinados['PIM'] - datos_combinados['PIM_Predicho']
            datos_combinados['Error_Absoluto_Soles'] = datos_combinados['Error_Soles'].abs().fillna(0)

            st.subheader("📌 Valores PIA Y PIM (Suma Total) por Año")
            años_referencia = {
                2019: {'PIA': 21349738, 'PIM': 35668999},
                2020: {'PIA': 19461375, 'PIM': 37380583},
                2021: {'PIA': 17086331, 'PIM': 47355258},
                2022: {'PIA': 32231260, 'PIM': 106494985},
                2023: {'PIA': 49927800, 'PIM': 90106666},
                2024: {'PIA': 59014885, 'PIM': 104240404}
            }

            años_presentes = datos_combinados['Año'].unique()
            cols = st.columns(3)
            col_idx = 0

            for año in sorted(años_presentes):
                if año in años_referencia:
                    with cols[col_idx % 3]:
                        st.markdown(f"""
                                    **Año {año}**  
                                    PIA: S/ {años_referencia[año]['PIA']:,}  
                                    PIM: S/ {años_referencia[año]['PIM']:,}
                                    """)
                    col_idx += 1

            umbral_error = 0.5 * datos_combinados['PIM'].abs().mean()
            outliers = datos_combinados[datos_combinados['Error_Absoluto_Soles'] > umbral_error]

            st.subheader("🔍 Resumen de Desviaciones")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PIM Real (Promedio)", f"S/ {datos_combinados['PIM'].mean():,.0f}")
            with col2:
                st.metric("PIM Predicho (Modelo)", f"S/ {datos_combinados['PIM_Predicho'].mean():,.0f}")
            with col3:
                st.metric("Desviación Promedio", f"S/ {datos_combinados['Error_Absoluto_Soles'].mean():,.0f}")

            if not outliers.empty:
                st.error(f"⚠️ Alertas: Se detectaron {len(outliers)} proyectos con desviaciones significativas.")
                with st.expander("📋 Detalles de proyectos con desviación"):
                    required_cols = ['Tipo_Dataset', 'Nombre_Original', 'PIA', 'PIM', 'PIM_Predicho',
                                     'Error_Soles']
                    available_cols = [col for col in required_cols if col in outliers.columns]
                    st.dataframe(
                        outliers[available_cols].sort_values('Error_Soles', ascending=False)
                        .style.format({
                            'PIA': 'S/ {:,.0f}',
                            'PIM': 'S/ {:,.0f}',
                            'PIM_Predicho': 'S/ {:,.0f}',
                            'Error_Soles': 'S/ {:,.0f}',
                        })
                    )
            else:
                st.success("✅ No se detectaron errores significativos en las predicciones")

            st.subheader("Recomendaciones")
            st.info("""
            **Para proyectos con alta desviación:**  
            🔹 **Error positivo (PIM > Predicho):** Fondos no utilizados eficientemente. 
            🔹 **Error negativo (PIM < Predicho):** Posible recursos no utilizados predictivamente.         

            **Pasos siguientes:**  
            🔹 Revisar los proyectos listados en alertas. 
            """)