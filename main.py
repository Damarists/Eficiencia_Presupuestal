from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

st.set_page_config(page_title="Analizador Presupuestario", page_icon="💰", layout="wide")

@st.cache_resource
def cargar_modelos():
    return joblib.load('modelos_presupuesto_unificados.pkl')

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
                            return float(value.replace('.', '', len(parts)-1).replace('.', '.'))
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
                        st.warning(f"Se encontraron {len(nuevas_categorias)} categorías no vistas durante el entrenamiento")
                        with st.expander("Ver categorías nuevas"):
                            st.write(list(nuevas_categorias)[:10]) 
                    
                    df['Nombre_encoded'] = df['Categoría_Presupuestal'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
                    )
                    df['Nombre_Original'] = df['Categoría_Presupuestal']
                
                elif 'Productos/Proyectos' in df.columns:
                    nuevos_proyectos = set(df['Productos/Proyectos']) - set(label_encoder.classes_)
                    if nuevos_proyectos:
                        st.warning(f"Se encontraron {len(nuevos_proyectos)} proyectos no vistos durante el entrenamiento")
                        with st.expander("Ver proyectos nuevos"):
                            st.write(list(nuevos_proyectos)[:10])
                    
                    df['Nombre_encoded'] = df['Productos/Proyectos'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
                    )
                    df['Nombre_Original'] = df['Productos/Proyectos']
                
                elif 'Función' in df.columns:
                    nuevas_funciones = set(df['Función']) - set(label_encoder.classes_)
                    if nuevas_funciones:
                        st.warning(f"Se encontraron {len(nuevas_funciones)} funciones no vistas durante el entrenamiento")
                        with st.expander("Ver funciones nuevas"):
                            st.write(list(nuevas_funciones)[:10])
                    
                    df['Nombre_encoded'] = df['Función'].apply(
                        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
                    )
                    df['Nombre_Original'] = df['Función']
                
                df['Tipo_Dataset_encoded'] = label_encoder.transform([tipo]*len(df))
                
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
            "🔮 Proyecciones",
            "🏛 Impacto Organizacional",
            "📝 Certificación",
            "🤝 Compromiso",
            "💰 Ejecución Devengado(Acumulado)",
            "💸 Girado"
        ])
        
        # 1. Optimización de Recursos
        with tab1:
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
                    
                    st.subheader("Resultados por Tipo de Datos")
                    
                    cols = st.columns(3)
                    for i, (tipo, df) in enumerate(datos_combinados.groupby('Tipo_Dataset')):
                        with cols[i]:
                            st.metric(label=f"{tipo} - Avance Real", value=f"{df['Avance %'].mean():.1f}%")
                            st.metric(label=f"{tipo} - Avance Esperado", value=f"{df['Avance_Predicho'].mean():.1f}%")
                            eficiencia = "✅ Buena" if df['Diferencia'].mean() >= 0 else "⚠️ Mejorable"
                            st.metric(label="Eficiencia", value=eficiencia)
                    
                    st.subheader("Recomendaciones")
                    if datos_combinados['Diferencia'].mean() < -5:
                        st.warning("""
                        **Oportunidad de mejora:**  
                        El avance real está por debajo de lo esperado en varios rubros.  
                        🔹 Revisar los proyectos con mayor diferencia negativa  
                        🔹 Evaluar posibles cuellos de botella en la ejecución  
                        🔹 Considerar redistribución de recursos a áreas más eficientes
                        """)
                    else:
                        st.success("""
                        **Buen desempeño:**  
                        La asignación de recursos está siendo eficiente en general.  
                        🔹 Mantener los procesos actuales  
                        🔹 Monitorear áreas con pequeñas diferencias  
                        🔹 Replicar buenas prácticas en otras áreas
                        """)
                    
                    with st.expander("📊 Ver detalles por item"):
                        st.dataframe(datos_combinados[['Tipo_Dataset', 'Nombre_Original', 'Avance %', 'Avance_Predicho', 'Diferencia']].sort_values('Diferencia'))
        
        # 2. Análisis y Monitoreo
        with tab2:
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
                        st.error(f"⚠️ Alertas: Se detectaron {len(outliers)} proyectos con desviacións significativas.")
                        with st.expander("📋 Detalles de proyectos con desviación"):
                            required_cols = ['Tipo_Dataset', 'Nombre_Original', 'PIA', 'PIM', 'PIM_Predicho', 'Error_Soles']
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
                    
                    🔹 **Error negativo (PIM < Predicho):** Posible subfinanciamiento.         
                    
                    **Pasos siguientes:**  
                    🔹 Revisar los proyectos listados en alertas. 
                    """)
        
        # 3. Toma de Decisiones Estratégicas
        with tab3:
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
                    y_proba = model.predict_proba(X)[:, 1]  # Probabilidad de éxito
                    
                    datos_combinados['Prob_Exito'] = y_proba * 100
                    datos_combinados['Recomendacion'] = np.where(y_pred == 1, "Priorizar", "Revisar")
                    
                    st.subheader("Recomendaciones Estratégicas")
                    
                    # Top 5 para priorizar
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
                    
                    st.subheader("Guía de Acción")
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
        
        # 4. Capacidad Tecnológica
        with tab4:
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
                            "Tiempo_Procesamiento":[len(categorias)*0.1, len(proyectos)*0.2, len(funcion)*0.3]
                        })

                    funcion = st.session_state.datos.get('funcion', pd.DataFrame())  
                    categorias = st.session_state.datos.get('categoria', pd.DataFrame())
                    proyectos = st.session_state.datos.get('proyectos', pd.DataFrame())
                    resumen = medir_capacidad_ingreso_datos(categorias, proyectos, funcion)
                    st.dataframe(resumen)
                    st.success("✅ Datos procesados exitosamente.")
                    
        # 5. Proyecciones
        with tab5:
            st.header("🔮 Proyecciones Presupuestarias")
            st.markdown("""
            **¿Qué analiza?**  
            Predice ejecuciones presupuestarias futuras utilizando Machine Learning para mejorar la planificación financiera.
            """)
    
            if st.button("Generar Proyecciones", key="proy_btn"):
                with st.spinner('Generando proyecciones presupuestarias...'):
                    model, scaler = modelos['proyecciones_presupuestarias']
                    X = datos_combinados[['PIA', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
                    X_scaled = scaler.transform(X)
                    predicciones = model.predict(X_scaled)
                    
                    columnas_pred = ['PIM_Predicho', 'Compromiso_Predicho', 'Certificacion_Predicha', 'Devengado_Predicho', 'Girado_Predicho', 'Avance_Porcentaje_Predicho']
        
                    predicciones_df = pd.DataFrame(predicciones, columns=columnas_pred)
                    predicciones_df['Tipo_Dataset'] = datos_combinados['Tipo_Dataset'].values
                    predicciones_df['Nombre_Original'] = datos_combinados['Nombre_Original'].values
                    predicciones_df['Año'] = datos_combinados['Año'].values
                    
                    predicciones_df = predicciones_df[['Tipo_Dataset', 'Nombre_Original', 'Año'] + columnas_pred]
                    
                    st.subheader("📋 Proyecciones Generadas")
                    st.dataframe(
                        predicciones_df.style.format({
                            'PIM_Predicho': 'S/ {:,.0f}',
                            'Compromiso_Predicho': 'S/ {:,.0f}',
                            'Certificacion_Predicha': 'S/ {:,.0f}',
                            'Devengado_Predicho': 'S/ {:,.0f}',
                            'Girado_Predicho': 'S/ {:,.0f}',
                            'Avance_Porcentaje_Predicho': '{:,.2f} %'
                        })
                    )
                    
                st.success("✅ Predicciones generadas correctamente.")

        # 6. Impacto Organizacional
        with tab6:
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
            
                    st.success("✅ Impacto organizacional evaluado")
                    
                    
        # 7. Certificación
        with tab7:
            st.header("📝 Certificación")
            st.markdown("""
            **¿Qué analiza?**  
            Detecta discrepancias en los procesos de certificación presupuestaria mediante Machine Learning.
            """)
    
            if st.button("Detectar Discrepancias en Certificación", key="cert_btn"):
                with st.spinner('Analizando discrepancias de certificación...'):
                    model = modelos['certificacion']
                    X = datos_combinados[['PIA', 'PIM', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
                    y_pred = model.predict(X)
            
                    datos_combinados['Discrepancia_Certificacion_Predicha'] = y_pred
                    
                    datos_combinados['Discrepancia_Real'] = (datos_combinados['Certificación'] - datos_combinados['PIM']).abs()
                           
                    st.subheader("Top 10 Discrepancias en Certificación")
                    st.dataframe(datos_combinados[['Tipo_Dataset', 'Nombre_Original', 'Certificación', 'PIM', 'Discrepancia_Real', 'Discrepancia_Certificacion_Predicha']].sort_values('Discrepancia_Certificacion_Predicha', ascending=False).head(10))
                    
                    top_discrepancias = datos_combinados[['Nombre_Original', 'Discrepancia_Certificacion_Predicha']].sort_values('Discrepancia_Certificacion_Predicha', ascending=False).head(10)
                    st.bar_chart(top_discrepancias.set_index('Nombre_Original')['Discrepancia_Certificacion_Predicha'])                    

        # 8. Compromiso
        with tab8:
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
            
                    datos_combinados['Porcentaje_Compromiso'] = (datos_combinados['Compromiso Anual'] / datos_combinados['PIM']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
                    datos_combinados['Porcentaje_Compromiso_Predicho'] = y_pred
                    
                    datos_combinados['Diferencia_Compromiso'] = (datos_combinados['Porcentaje_Compromiso'] - datos_combinados['Porcentaje_Compromiso_Predicho']).abs()
            
                    def clasificar_transparencia(row):
                        if row['Diferencia_Compromiso'] <= 5:
                            return 'Alta'
                        elif row['Diferencia_Compromiso'] <= 15:
                            return 'Media'
                        else:
                            return 'Baja'
            
                    datos_combinados['Transparencia'] = datos_combinados.apply(clasificar_transparencia, axis=1)
            
                    st.subheader("Resultados del Monitoreo de Compromisos")
                    st.dataframe(datos_combinados[['Tipo_Dataset', 'Nombre_Original', 'Porcentaje_Compromiso_Predicho', 'Porcentaje_Compromiso', 'Diferencia_Compromiso', 'Transparencia']].sort_values('Porcentaje_Compromiso_Predicho', ascending=False))

                    st.subheader("Comparación entre porcentaje de compromiso real y predicho")
                    top_compromisos = datos_combinados[['Nombre_Original', 'Porcentaje_Compromiso', 'Porcentaje_Compromiso_Predicho']].sort_values('Porcentaje_Compromiso_Predicho', ascending=False).head(10)
                    st.bar_chart(top_compromisos.set_index('Nombre_Original')[['Porcentaje_Compromiso', 'Porcentaje_Compromiso_Predicho']])

                    st.success("✅ Monitoreo de compromisos completado")


        # 9. Devengado
        with tab9:
            st.header("💰 Ejecución Devengado")
            st.markdown("""
            **¿Qué analiza?**  
            Detecta errores automáticamente en los registros financieros de ejecución de devengado y sugiere posibles oportunidades de mejora en el proceso de devengado presupuestal utilizando Machine Learning.
            """)

            if st.button("Detectar Errores de Devengado", key="dev_btn"):
                with st.spinner('Detectando errores en devengado...'):
                    model = modelos['ejecucion_devengado']

                    X = datos_combinados[['PIA', 'PIM', 'Compromiso Anual', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
                    y_pred = model.predict(X)
            
                    datos_combinados['Alerta_Devengado'] = np.where(y_pred == 1, "⚠️ Error detectado", "✅ Correcto")
                    errores = datos_combinados[datos_combinados['Alerta_Devengado'] == "⚠️ Error detectado"]

                    st.subheader("Errores Detectados en Ejecución de Devengado")
            
                    if not errores.empty:
                        st.error(f"⚠️ Se detectaron {len(errores)} posibles errores en los registros de ejecución de devengado.")
                        with st.expander("📋 Ver detalles de errores detectados"):
                            st.dataframe(errores[['Tipo_Dataset', 'Nombre_Original', 'Compromiso Anual', 'PIA', 'PIM', 'Ejecución_Devengado', 'Alerta_Devengado']])
                            st.markdown("""
                            **¿Por qué podría haberse marcado un error?**  
                            Si un error se detecta en el devengado, es posible que exista una **discrepancia significativa** entre la ejecución del gasto (Ejecución_Devengado) y el compromiso presupuestario (Compromiso Anual). Esto podría deberse a una **sobrecarga o subejecución** en el gasto, que es un indicio de que el proceso de devengado no está funcionando correctamente.
                            A continuación, puedes ver las celdas que contribuyeron a la detección de este error.
                            """)

                            st.markdown("### Detalles de las celdas que contribuyeron al error de ejecución de devengado:")
                            for index, row in errores.iterrows():
                                st.markdown(f"#### Fila {index}:")
                                st.write(row[['PIA', 'PIM', 'Compromiso Anual', 'Año', 'Ejecución_Devengado', 'Alerta_Devengado']])
            
                    else:
                        st.success("✅ No se detectaron errores significativos en la ejecución de devengado.")

                    st.subheader("Importancia de las características para la detección de errores")
                    feature_importances = model.feature_importances_
                    feature_names = X.columns
                    feature_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances
                    }).sort_values(by='Importance', ascending=False)

                    st.bar_chart(feature_df.set_index('Feature')['Importance'])

                    st.success("✅ Análisis de ejecución de devengado completado")

 
        
        # 10. Ejecución Girado
        with tab10:
            st.header("💸 Girado")
            st.markdown("""
            **¿Qué analiza?**  
            Detecta discrepancias entre lo devengado y lo girado para prevenir errores en desembolsos. Este análisis se centra en las diferencias entre la **Ejecución de Girado** y la **Ejecución de Devengado** para identificar posibles problemas o errores en el proceso de giros.
            """)

            if st.button("Analizar Girados", key="gir_btn"):
                with st.spinner('Buscando discrepancias...'):
                    model = modelos['ejecucion_girado']

                    X = datos_combinados[['PIA', 'PIM', 'Ejecución_Devengado', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
                    y_pred = model.predict(X)
            
                    datos_combinados['Alerta_Girado'] = np.where(y_pred == 1, "⚠️ Revisar", "✅ Correcto")
                    discrepancias = datos_combinados[datos_combinados['Alerta_Girado'] == "⚠️ Revisar"]

                    st.subheader("Resultados del Análisis de Girado")
            
                    cols = st.columns(3)
                    for i, (tipo, df) in enumerate(datos_combinados.groupby('Tipo_Dataset')):
                        with cols[i]:
                            total = len(df)
                            problemas = len(df[df['Alerta_Girado'] == "⚠️ Revisar"])
                            st.metric(label=f"{tipo} - Items con problemas", value=f"{problemas} de {total}")
            
                    if not discrepancias.empty:
                        st.error(f"⚠️ Se detectaron {len(discrepancias)} posibles discrepancias que requieren revisión")
                        with st.expander("📋 Ver detalles de discrepancias"):
                            st.dataframe(discrepancias[['Tipo_Dataset', 'Nombre_Original', 'Ejecución_Devengado', 'Ejecución_Girado', 'Discrepancia_Girado']])
                    
                            st.markdown("""
                            **¿Por qué podría haberse marcado una discrepancia?**  
                            Una discrepancia significativa entre la **Ejecución Girado** y la **Ejecución Devengado** puede indicar que se ha girado más o menos dinero de lo que realmente se ha ejecutado en los compromisos. Esto puede ser un indicio de errores administrativos, pagos duplicados, o falta de conciliación de las cuentas.
                            A continuación, se presentan las celdas clave que contribuyeron a la discrepancia detectada.
                            """)

                            for index, row in discrepancias.iterrows():
                                st.markdown(f"#### Fila {index}:")
                                st.write(row[['PIA', 'PIM', 'Ejecución_Devengado', 'Ejecución_Girado', 'Discrepancia_Girado']])
            
                    else:
                        st.success("✅ No se detectaron discrepancias significativas en los girados.")

                    st.subheader("Importancia de las características en la detección de discrepancias")
                    feature_importances = model.feature_importances_
                    feature_names = X.columns
                    feature_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances
                    }).sort_values(by='Importance', ascending=False)

                    st.bar_chart(feature_df.set_index('Feature')['Importance'])

                    st.success("✅ Análisis de ejecución de girado completado")

else:
    st.info("👈 Por favor sube los archivos CSV en la barra lateral para comenzar el análisis")

# Footer
st.markdown("---")
st.markdown("© 2025 Sistema Inteligente de Análisis Presupuestario - Versión 1.0")