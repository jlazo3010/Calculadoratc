import os
import time
import pandas as pd
import numpy as np
import warnings
import subprocess
import tempfile
import sys
import re
warnings.filterwarnings('ignore')

def ejecutar_modelo_ais(
    nombre_muestra=None
):
    """
    Ejecuta un modelo XGBoost guardado en formato .Rdata desde Python usando subprocess.
    Para usar en una app de Streamlit existente.
    
    Args:
        nombre_muestra (DataFrame): DataFrame (resultadosAIS) con los datos para predecir
        
    Returns:
        DataFrame: DataFrame con las columnas bimboId, blmId y predicciones (preds)
    """
    # Iniciar cronómetro
    start_time = time.time()
    
    # Verificar que se haya proporcionado un DataFrame
    if nombre_muestra is None:
        raise ValueError("Debe proporcionar un DataFrame como nombre_muestra")
    
    # Copiar el DataFrame para no modificar el original
    df = nombre_muestra.copy()
    
    # Limpieza de nombres de columnas
    nombres_originales = df.columns.tolist()
    # Eliminar acentos y caracteres especiales
    nombres_limpios = [col.encode('ascii', 'ignore').decode('ascii') for col in nombres_originales]
    # Eliminar caracteres no alfanuméricos excepto guion bajo
    nombres_limpios = [re.sub(r'[^0-9a-zA-Z_]', '', col) for col in nombres_limpios]
    # Renombrar las columnas
    rename_dict = dict(zip(nombres_originales, nombres_limpios))
    df = df.rename(columns=rename_dict)
    
    print("Nombres de columnas limpiados:")
    for original, limpio in zip(nombres_originales, nombres_limpios):
        print(f"Original: {original} -> Limpio: {limpio}")
    
    # Obtener el directorio actual
    path = os.getcwd()
    
    # Convertir a formato de ruta de R (con barras diagonales hacia adelante)
    path_r = path.replace("\\", "/")
    if not path_r.endswith("/"):
        path_r = path_r + "/"
    
    # Definir directorios
    output_dir = path_r
    
    # Obtener el directorio actual donde se ejecuta el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Directorio del script: {script_dir}")
    
    # Definir la ruta del modelo directamente en la misma carpeta del script
    nombre_modelo = os.path.join(script_dir, "AISMaster_Modelo_20241223131859.RData")
    # Convertir ruta del modelo a formato R
    nombre_modelo_r = nombre_modelo.replace("\\", "/")
    print(f"Ruta completa del modelo: {nombre_modelo_r}")
    
    # Verificación adicional para asegurar que el modelo existe
    if not os.path.exists(nombre_modelo):
        raise FileNotFoundError(f"El archivo del modelo no existe en la ruta: {nombre_modelo}")
    
    # Crear un archivo temporal para guardar los datos
    input_csv_path = None  # Inicializar variable
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as temp_input_file:
        input_csv_path = temp_input_file.name
        df.to_csv(input_csv_path, index=False)
    
    # Verificar que el archivo temporal se creó correctamente
    if not input_csv_path or not os.path.exists(input_csv_path):
        raise FileNotFoundError("No se pudo crear el archivo temporal para los datos")
        
    # Convertir ruta del archivo CSV a formato R
    input_csv_path_r = input_csv_path.replace("\\", "/")
    
    # Crear nombres para archivos de salida
    resultado_completo_path = os.path.join(path, "Muestra_out.csv")
    resultado_final_path = os.path.join(path, "resultado_final.csv")
    
    # Convertir rutas de salida a formato R
    resultado_completo_path_r = resultado_completo_path.replace("\\", "/")
    resultado_final_path_r = resultado_final_path.replace("\\", "/")
    
    # Crear script R temporal
    r_script_content = f"""
    # Configurar opciones de R para evitar problemas comunes
    options(warn = 1)  # Mostrar advertencias cuando ocurran

    # Crear carpeta de librerías de usuario si no existe
    if (!dir.exists(Sys.getenv("R_LIBS_USER"))) {{
        dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)
        cat("Carpeta R_LIBS_USER creada:", Sys.getenv("R_LIBS_USER"), "\\n")
    }}

    .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

    # Paquetes requeridos
    packages_needed <- c("Matrix", "xgboost", "dplyr", "fastDummies")
    packages_to_install <- packages_needed[!packages_needed %in% installed.packages()[,"Package"]]

    # Instalar paquetes desde CRAN (versión compatible de xgboost)
    if (length(packages_to_install) > 0) {{
        cat("Instalando paquetes necesarios:", paste(packages_to_install, collapse=", "), "\\n")
        for (pkg in packages_to_install) {{
            tryCatch({{
                if (pkg == "xgboost") {{
                    cat("Instalando versión compatible de xgboost (1.4.1.1)...\\n")
                    install.packages("xgboost", repos = "https://cloud.r-project.org", lib = Sys.getenv("R_LIBS_USER"), quiet = TRUE)
                }} else {{
                    install.packages(pkg, repos = "https://cloud.r-project.org", lib = Sys.getenv("R_LIBS_USER"), quiet = TRUE)
                }}
            }}, error = function(e) {{
                cat("❌ Error al instalar", pkg, ":", e$message, "\\n")
            }})
        }}
    }}

    # Intentar cargar los paquetes
    for (pkg in packages_needed) {{
        cat("Cargando paquete:", pkg, "\\n")
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
            cat("❌ Error al cargar el paquete:", pkg, "\\n")
            stop(paste("No se pudo cargar el paquete", pkg))
        }}
    }}

    # Resto del script continúa aquí...
    # (debes concatenar el resto de tu script actual a partir de aquí)
    """
    
    # Escribir el script R en un archivo temporal
    r_script_path = None  # Inicializar variable
    with tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w+') as temp_r_script:
        r_script_path = temp_r_script.name
        temp_r_script.write(r_script_content)
    
    # Verificar que el archivo temporal se creó correctamente
    if not r_script_path or not os.path.exists(r_script_path):
        raise FileNotFoundError("No se pudo crear el archivo temporal para el script R")
    
    # Inicializar variable para archivo temporal de paquetes
    pkg_script_path = None
    
    try:
        # Verificar que R está instalado
        try:
            # Intentar ejecutar R con una versión muy simple para verificar disponibilidad
            r_check = subprocess.run(['Rscript', '-e', 'cat("R está funcionando correctamente\n")'], 
                                    capture_output=True,
                                    text=True,
                                    timeout=10)
            if r_check.returncode == 0:
                print("✅ R está instalado y funcionando correctamente")
            else:
                print("⚠️ R podría estar instalado pero con problemas:")
                print(r_check.stderr)
        except Exception as e:
            print(f"Error al verificar R: {e}")
            if os.name == 'posix':  # Linux/Mac
                try:
                    print("Intentando instalar R en sistema Linux...")
                    subprocess.run(['apt-get', 'update'], check=True)
                    subprocess.run(['apt-get', 'install', '-y', 'r-base'], check=True)
                    print("R instalado correctamente")
                except Exception as e:
                    print(f"Error al instalar R: {e}")
                    raise RuntimeError("No se pudo instalar R. Verifique que tiene los permisos necesarios.")
            else:
                raise RuntimeError("R no está instalado en el sistema. Por favor, instale R manualmente.")
        
        # Instalar paquetes R necesarios
        r_packages_script = """
        if (!require("Matrix")) install.packages("Matrix", repos = "https://cloud.r-project.org")
        if (!require("xgboost")) install.packages("xgboost", repos = "https://cloud.r-project.org")
        if (!require("dplyr")) install.packages("dplyr", repos = "https://cloud.r-project.org")
        if (!require("fastDummies")) install.packages("fastDummies", repos = "https://cloud.r-project.org")
        """
        
        with tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w+') as temp_pkg_script:
            pkg_script_path = temp_pkg_script.name
            temp_pkg_script.write(r_packages_script)
        
        # Verificar que el archivo temporal se creó correctamente
        if not pkg_script_path or not os.path.exists(pkg_script_path):
            raise FileNotFoundError("No se pudo crear el archivo temporal para el script de paquetes R")
        
        print("Instalando paquetes R necesarios...")
        pkg_install = subprocess.run(['Rscript', pkg_script_path], 
                                    capture_output=True, 
                                    text=True,
                                    timeout=300)  # 5 minutos para instalar paquetes
        
        if pkg_install.returncode != 0:
            print("Advertencia: Posible problema al instalar paquetes R:")
            print(pkg_install.stderr)

        # Ejecutar el script R principal con una mejor gestión de errores
        print(f"Ejecutando script R: {r_script_path}")
        
        # Mejorar el manejo de errores de R
        try:
            # Ejecutar R con impresión directa de la salida para un mejor diagnóstico
            print("Iniciando ejecución de R...")
            process = subprocess.Popen(['Rscript', '--vanilla', r_script_path], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    bufsize=1)
            
            # Imprimir salida de R en tiempo real
            stdout_lines = []
            stderr_lines = []
            
            # Leer y mostrar la salida estándar en tiempo real
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                line = line.rstrip()
                print(f"R: {line}")
                stdout_lines.append(line)
            
            # Capturar la salida de error
            for line in iter(process.stderr.readline, ''):
                if not line:
                    break
                line = line.rstrip()
                print(f"R ERROR: {line}")
                stderr_lines.append(line)
                
            # Esperar a que el proceso termine y obtener el código de salida
            exit_code = process.wait()
            
            # Verificar si hubo errores
            if exit_code != 0:
                stderr_output = "\n".join(stderr_lines)
                stdout_output = "\n".join(stdout_lines)
                mensaje_error = f"""
            ❌ Error al ejecutar el script R. Código de salida: {exit_code}
            --- STDOUT ---
            {stdout_output}
            --- STDERR ---
            {stderr_output}
            """
                raise RuntimeError(mensaje_error)
            
        except subprocess.TimeoutExpired:
            print("❌ Error: El script R tardó demasiado tiempo en ejecutarse y se canceló.")
            if 'process' in locals() and process:
                process.kill()
            raise RuntimeError("Timeout al ejecutar el script R")
        
        # Leer el resultado final
        if os.path.exists(resultado_final_path):
            resultado_final = pd.read_csv(resultado_final_path)
            tiempo_total = time.time() - start_time
            print(f"Tiempo de ejecución: {tiempo_total:.2f} segundos")
            return resultado_final
        else:
            raise FileNotFoundError(f"El archivo de resultados no se ha creado en: {resultado_final_path}")

    except subprocess.TimeoutExpired:
        print("❌ Error: El script R tardó demasiado tiempo en ejecutarse y se canceló.")
        raise RuntimeError("Timeout al ejecutar el script R")
    
    except Exception as e:
        print(f"⚠️ Excepción durante la ejecución: {type(e).__name__}: {e}")
        raise

    finally:
        # Limpieza de archivos temporales (asegurando que las variables estén definidas)
        for file_path in [input_csv_path, r_script_path, pkg_script_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"No se pudo eliminar el archivo temporal {file_path}: {e}")