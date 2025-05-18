import streamlit as st
import pandas as pd
import numpy as np

def mostrar(modelos, datos_combinados):
    st.header("🎯 Toma de Decisiones Estratégicas")
    st.markdown("""
    **¿Qué analiza?**  
    Identifica qué proyectos o categorías tienen mayor probabilidad de éxito para priorizar recursos.
    """)

    if st.button("Evaluar Decisiones", key="dec_btn"):
        with st.spinner('Analizando viabilidad...'):
            model = modelos['decisiones_estrategicas']
            X = datos_combinados[['PIA', 'PIM', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            datos_combinados['Prob_Exito'] = y_proba * 100
            datos_combinados['Recomendacion'] = np.where(y_pred == 1, "Priorizar", "Revisar")

            st.subheader("Recomendaciones Estratégicas")
            top_priorizar = datos_combinados.sort_values('Prob_Exito', ascending=False).head(5)

            cols = st.columns(2)
            with cols[0]:
                st.metric("Items recomendados para priorizar",
                          f"{len(datos_combinados[datos_combinados['Recomendacion'] == 'Priorizar'])}")
            with cols[1]:
                st.metric("Items que requieren revisión",
                          f"{len(datos_combinados[datos_combinados['Recomendacion'] == 'Revisar'])}")

            st.write("**Top 5 items recomendados para priorizar:**")
            st.dataframe(top_priorizar[['Tipo_Dataset', 'Nombre_Original', 'Prob_Exito', 'Recomendacion']])

            st.subheader("Todos los items recomendados para priorizar")
            items_priorizar = datos_combinados[datos_combinados['Recomendacion'] == 'Priorizar']
            st.dataframe(items_priorizar[['Tipo_Dataset', 'Nombre_Original', 'Prob_Exito', 'Recomendacion']])

            st.subheader("Todos los items recomendados para revisar")
            items_priorizar = datos_combinados[datos_combinados['Recomendacion'] == 'Revisar']
            st.dataframe(items_priorizar[['Tipo_Dataset', 'Nombre_Original', 'Prob_Exito', 'Recomendacion']])

            st.subheader("Guía de Acción")
            st.info("""**Nota:**
            Los que se encuentran en la sección de Priorizar son los que primero han pasado por un filtro en el que su avance de ejecución ha sido mayor a 80% y los de revisión su avance de ejecución es menor a 80%.
            """)
            st.success("""
            **Para items 'Priorizar':**  
            🔹 Asignar recursos según lo planeado  
            🔹 Mantener seguimiento estándar  
            🔹 Replicar buenas prácticas
            """)

            st.warning("""
            **Para items 'Revisar':**  
            🔹 Analizar causas de bajo desempeño esperado  
            🔹 Considerar ajustes en asignación  
            🔹 Implementar planes de contingencia  
            🔹 Aumentar frecuencia de monitoreo
            """)