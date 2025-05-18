import streamlit as st
import pandas as pd
import numpy as np

def mostrar(modelos, datos_combinados):
    st.header("💸 Girado")
    st.markdown("""
    **¿Qué analiza?**  
    Detecta discrepancias entre lo devengado y lo girado para prevenir errores en desembolsos. Este análisis se centra en las diferencias entre la **Ejecución de Girado** y la **Ejecución de Devengado** para identificar posibles problemas o errores en el proceso de giros.
    """)

    if st.button("Analizar Girados", key="gir_btn"):
        with st.spinner('Buscando discrepancias...'):
            model = modelos['ejecucion_girado']
            X = datos_combinados[
                ['PIA', 'PIM', 'Ejecución_Devengado', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
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
                    st.dataframe(discrepancias[['Tipo_Dataset', 'Nombre_Original', 'Ejecución_Devengado',
                                                'Ejecución_Girado', 'Discrepancia_Girado']])

                    st.markdown("""
                    **¿Por qué podría haberse marcado una discrepancia?**  
                    Una discrepancia significativa entre la **Ejecución Girado** y la **Ejecución Devengado** puede indicar que se ha girado más o menos dinero de lo que realmente se ha ejecutado en los compromisos. Esto puede ser un indicio de errores administrativos, pagos duplicados, o falta de conciliación de las cuentas.
                    A continuación, se presentan las celdas clave que contribuyeron a la discrepancia detectada.
                    """)

                    for index, row in discrepancias.iterrows():
                        st.markdown(f"#### Fila {index}:")
                        st.write(row[['PIA', 'PIM', 'Ejecución_Devengado', 'Ejecución_Girado',
                                      'Discrepancia_Girado']])
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