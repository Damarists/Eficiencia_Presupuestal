import streamlit as st
import pandas as pd

def mostrar(modelos, datos_combinados):
    st.header("🧮 Capacidad de Procesamiento")
    st.markdown("""
    **¿Qué analiza?**  
    Estima la capacidad de procesamiento de datos en tiempo real de Machine Learning.
    """)

    if st.button("Evaluar Capacidad", key="capacidad_btn"):
        with st.spinner("Procesando..."):
            def medir_capacidad_ingreso_datos(categorias, proyectos, funcion):
                return pd.DataFrame({
                    "Tipo_Dataset": ["Categoría", "Proyecto", "Función"],
                    "Carga_Procesamiento": [len(categorias), len(proyectos), len(funcion)],
                    "Tiempo_Procesamiento": [len(categorias) * 0.1, len(proyectos) * 0.2, len(funcion) * 0.3]
                })

            funcion = st.session_state.datos.get('funcion', pd.DataFrame())
            categorias = st.session_state.datos.get('categoria', pd.DataFrame())
            proyectos = st.session_state.datos.get('proyectos', pd.DataFrame())
            resumen = medir_capacidad_ingreso_datos(categorias, proyectos, funcion)
            st.dataframe(resumen)
            st.success("✅ Datos procesados exitosamente.")