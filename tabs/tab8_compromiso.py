import streamlit as st
import pandas as pd
import numpy as np

def mostrar(modelos, datos_combinados):
    st.header("🤝 Compromiso")
    st.markdown("""
    **¿Qué analiza?**  
    Mide el porcentaje de compromisos presupuestales monitoreados y la transparencia en su ejecución usando Machine Learning.
    """)

    if st.button("Monitorear Compromisos", key="comp_btn"):
        with st.spinner('Monitoreando compromisos...'):
            model = modelos['compromiso']
            X = datos_combinados[['PIA', 'PIM', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            y_pred = model.predict(X)

            datos_combinados['Porcentaje_Compromiso'] = (datos_combinados['Compromiso Anual'] /
                                                         datos_combinados['PIM']).replace([np.inf, -np.inf],
                                                                                          np.nan).fillna(0) * 100
            datos_combinados['Porcentaje_Compromiso_Predicho'] = y_pred
            datos_combinados['Diferencia_Compromiso'] = (
                datos_combinados['Porcentaje_Compromiso'] - datos_combinados['Porcentaje_Compromiso_Predicho']).abs()

            def clasificar_transparencia(row):
                if row['Diferencia_Compromiso'] <= 5:
                    return 'Alta'
                elif row['Diferencia_Compromiso'] <= 15:
                    return 'Media'
                else:
                    return 'Baja'

            datos_combinados['Transparencia'] = datos_combinados.apply(clasificar_transparencia, axis=1)

            st.subheader("Resultados del Monitoreo de Compromisos")
            st.dataframe(datos_combinados[['Tipo_Dataset', 'Nombre_Original', 'Porcentaje_Compromiso_Predicho',
                                           'Porcentaje_Compromiso', 'Diferencia_Compromiso',
                                           'Transparencia']].sort_values('Porcentaje_Compromiso_Predicho',
                                                                         ascending=False))

            st.subheader("Comparación entre porcentaje de compromiso real y predicho")
            top_compromisos = datos_combinados[
                ['Nombre_Original', 'Porcentaje_Compromiso', 'Porcentaje_Compromiso_Predicho']].sort_values(
                'Porcentaje_Compromiso_Predicho', ascending=False).head(10)
            st.bar_chart(top_compromisos.set_index('Nombre_Original')[
                ['Porcentaje_Compromiso', 'Porcentaje_Compromiso_Predicho']])

            st.subheader("Recomendaciones")
            st.info("""
            🔹 Si la Diferencia de Compromiso es menor e igual que 5%: Es Alta
            🔹 Si la Diferencia de Compromiso es menor e igual que 15%: Es Media
            🔹 Si la Diferencia de Compromiso es mayor que 15%: Es Baja
            """)

            st.success("✅ Monitoreo de compromisos completado")