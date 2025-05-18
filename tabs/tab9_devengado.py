import streamlit as st
import pandas as pd
import numpy as np

def mostrar(modelos, datos_combinados):
    st.header("💰 Ejecución Devengado")
    st.markdown("""
    **¿Qué analiza?**  
    Detecta errores automáticamente en los registros financieros de ejecución de devengado y sugiere posibles oportunidades de mejora en el proceso de devengado presupuestal utilizando Machine Learning.
    """)

    if st.button("Detectar Errores de Devengado", key="dev_btn"):
        with st.spinner('Detectando errores en devengado...'):
            model = modelos['ejecucion_devengado']
            X = datos_combinados[
                ['PIA', 'PIM', 'Compromiso Anual', 'Año', 'Tipo_Dataset_encoded', 'Nombre_encoded']]
            y_pred = model.predict(X)

            datos_combinados['Alerta_Devengado'] = np.where(y_pred == 1, "⚠️ Error detectado", "✅ Correcto")
            errores = datos_combinados[datos_combinados['Alerta_Devengado'] == "⚠️ Error detectado"]

            st.subheader("Errores Detectados en Ejecución de Devengado")

            if not errores.empty:
                st.error(
                    f"⚠️ Se detectaron {len(errores)} posibles errores en los registros de ejecución de devengado.")
                with st.expander("📋 Ver detalles de errores detectados"):
                    st.dataframe(errores[['Tipo_Dataset', 'Nombre_Original', 'Compromiso Anual', 'PIA', 'PIM',
                                          'Ejecución_Devengado', 'Alerta_Devengado']])
                    st.markdown("""
                    **¿Por qué podría haberse marcado un error?**  
                    Si un error se detecta en el devengado, es posible que exista una **discrepancia significativa** entre la ejecución del gasto (Ejecución_Devengado) y el compromiso presupuestario (Compromiso Anual). Esto podría deberse a una **sobrecarga o subejecución** en el gasto, que es un indicio de que el proceso de devengado no está funcionando correctamente.
                    A continuación, puedes ver las celdas que contribuyeron a la detección de este error.
                    """)

                    st.markdown("### Detalles de las celdas que contribuyeron al error de ejecución de devengado:")
                    for index, row in errores.iterrows():
                        st.markdown(f"#### Fila {index}:")
                        st.write(row[['PIA', 'PIM', 'Compromiso Anual', 'Año', 'Ejecución_Devengado',
                                      'Alerta_Devengado']])
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