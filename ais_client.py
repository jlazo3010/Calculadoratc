import pandas as pd
import requests

def predecir_ais_api(
    df, 
    modelo="AISMaster_Modelo_20241223131859.Rdata",
    api_url="https://f2b7-200-94-61-42.ngrok-free.app/predict"  # IP correcta del servidor Windows
):
    """
    Función cliente para llamar a la API del modelo AIS en un servidor Windows
    
    Args:
        df (DataFrame): DataFrame con los datos para predecir
        modelo (str, optional): Nombre del modelo a utilizar
        api_url (str): URL del servidor API
        
    Returns:
        DataFrame: DataFrame con las predicciones
    """
    try:
        # Limpiar valores no válidos
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        
        # Asegurar que los NaN se transformen correctamente a None para JSON
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        
        # Convertir a JSON-compatible
        json_data = df_clean.to_dict(orient="records")
        
        # Preparar payload para la API
        payload = {
            "modelo": modelo,
            "data": json_data
        }
        
        # Enviar solicitud
        response = requests.post(api_url, json=payload)
        
        # Procesar respuesta
        if response.status_code == 200:
            result = response.json()
            
            if result['status'] == 'success':
                predictions_df = pd.DataFrame(result['predictions'])
                return predictions_df
            else:
                raise Exception(f"Error en la API: {result['message']}")
        else:
            raise Exception(f"Error HTTP: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"Error al llamar a la API: {str(e)}")
        raise

# Ejemplo de uso en tu aplicación Streamlit
"""
import streamlit as st
import pandas as pd
from ais_client import predecir_ais_api  # Importa la función cliente

# Resto de tu código Streamlit...

# Cuando necesites ejecutar el modelo AIS
if st.button("Ejecutar modelo AIS"):
    try:
        # Supongamos que ya tienes un DataFrame llamado 'datos'
        resultados = predecir_ais_api(
            df=datos,
            modelo="AISMaster_Modelo_20241223131859.Rdata", 
            api_url="http://172.18.10.49:5000/predict"  # IP correcta del servidor
        )
        
        # Mostrar resultados
        st.success("¡Predicciones completadas!")
        st.dataframe(resultados)
        
    except Exception as e:
        st.error(f"Error al ejecutar el modelo AIS: {str(e)}")
"""
