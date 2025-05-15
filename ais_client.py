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
        # Convertir DataFrame a formato JSON
        json_data = df.replace([np.inf, -np.inf], np.nan).fillna(None).to_dict(orient="records")
        
        # Preparar datos para la solicitud
        payload = {
            "modelo": modelo,
            "data": json_data
        }
        
        # Realizar la solicitud a la API
        response = requests.post(api_url, json=payload)
        
        # Verificar si la solicitud fue exitosa
        if response.status_code == 200:
            result = response.json()
            
            # Verificar el estado de la respuesta
            if result['status'] == 'success':
                # Convertir las predicciones JSON a DataFrame
                predictions_json = result['predictions']
                predictions_df = pd.DataFrame(predictions_json)
                return predictions_df
            else:
                raise Exception(f"Error en la API: {result['message']}")
        else:
            raise Exception(f"Error en la solicitud: {response.status_code} - {response.text}")
    
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
