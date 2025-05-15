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
    nombre_modelo="AISMaster_Modelo_20241223131859.Rdata",
    nombre_muestra=None
):
    """
    Ejecuta un modelo XGBoost guardado en formato .Rdata desde Python usando subprocess.
    Para usar en una app de Streamlit existente.
    
    Args:
        nombre_modelo (str): Nombre o ruta al archivo del modelo en formato .Rdata
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
    
    # Verificar si el archivo del modelo existe
    if not os.path.isabs(nombre_modelo):
        # Si es una ruta relativa, hacerla absoluta
        nombre_modelo = os.path.join(path, nombre_modelo)
    
    if not os.path.exists(nombre_modelo):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {nombre_modelo}")
    
    # Convertir ruta del modelo a formato R
    nombre_modelo_r = nombre_modelo.replace("\\", "/")
    
    # Crear un archivo temporal para guardar los datos
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as temp_input_file:
        input_csv_path = temp_input_file.name
        df.to_csv(input_csv_path, index=False)
    
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
    # Cargar paquetes necesarios
    tryCatch({{
        library(Matrix)
        library(xgboost)
        library(dplyr)
        library(fastDummies)
    }}, error = function(e) {{
        # Si hay error al cargar los paquetes, instalamos
        cat("Error al cargar paquetes, intentando instalar:", e$message, "\\n")
        install.packages(c("Matrix", "xgboost", "dplyr", "fastDummies"), repos = "https://cloud.r-project.org")
        library(Matrix)
        library(xgboost)
        library(dplyr)
        library(fastDummies)
    }})
    
    # Debugging - Mostrar directorio de trabajo actual
    cat("Directorio de trabajo:", getwd(), "\\n")
    
    # Leer los datos
    tryCatch({{
        df_python <- read.csv("{input_csv_path_r}")
        cat("Datos cargados exitosamente, dimensiones:", dim(df_python)[1], "x", dim(df_python)[2], "\\n")
    }}, error = function(e) {{
        cat("Error al leer el archivo CSV:", e$message, "\\n")
        cat("Ruta del archivo:", "{input_csv_path_r}", "\\n")
        stop("No se pudo leer el archivo de entrada")
    }})
    
    # Verificar si existen las columnas necesarias
    columnas_requeridas <- c("PE_TC_PE_Ventas_AG", "PE_TC_PE_MUNICIPIO_AGR2", "PE_TC_PE_ENTIDAD_AGR")
    columnas_faltantes <- columnas_requeridas[!columnas_requeridas %in% names(df_python)]
    
    if (length(columnas_faltantes) > 0) {{
        cat("Advertencia: Algunas columnas requeridas no existen:\\n")
        cat(paste(columnas_faltantes, collapse=", "), "\\n")
        
        # Crear las columnas que faltan
        for (col in columnas_faltantes) {{
            df_python[[col]] <- NA
            cat("Columna creada:", col, "\\n")
        }}
    }}
    
    # Columnas disponibles para dummies
    columnas_dummy <- columnas_requeridas[columnas_requeridas %in% names(df_python)]
    
    # Conversión a dummies solo si hay columnas para convertir
    if (length(columnas_dummy) > 0) {{
        df <- fastDummies::dummy_cols(df_python, 
                                      select_columns = columnas_dummy,
                                      remove_selected_columns = TRUE)
    }} else {{
        df <- df_python
    }}
    
    # La conversión a dummies genera variables que tienen un guion bajo que no es necesario, hay que eliminarlo
    nombres_dummies <- grep("PE_TC_PE_Ventas_AG_|PE_TC_PE_MUNICIPIO_AGR2_|PE_TC_PE_ENTIDAD_AGR_", names(df), value = TRUE)
    
    # Renombrar solo las columnas dummies identificadas
    if (length(nombres_dummies) > 0) {{
        colnames(df)[colnames(df) %in% nombres_dummies] <- gsub("_(?!.*_)", "", colnames(df)[colnames(df) %in% nombres_dummies], perl = TRUE)
    }}
    
    # Cargar el modelo
    tryCatch({{
        cat("Intentando cargar el modelo desde:", "{nombre_modelo_r}", "\\n")
        load("{nombre_modelo_r}")
        cat("Modelo cargado exitosamente\\n")
    }}, error = function(e) {{
        cat("Error al cargar el modelo:", e$message, "\\n")
        stop("No se pudo cargar el modelo")
    }})
    
    # Selección de variables
    var.mod.xgboost <- AISMasterModelo$modelo$feature_names
    cat("Variables del modelo:", length(var.mod.xgboost), "\\n")
    
    # Asegurar que todas las variables del modelo estén en df
    for (var in var.mod.xgboost) {{
        if (!(var %in% colnames(df))) {{
            df[[var]] <- NA
            cat("Variable añadida:", var, "\\n")
        }}
    }}
    
    # Reemplazar NaN e Inf por NA
    df <- data.frame(lapply(df, function(x) {{
        replace(x, is.nan(x) | is.infinite(x), NA)
    }}))
    
    # Predicción
    tryCatch({{
        muestra.matrix <- xgboost::xgb.DMatrix(as.matrix(df[, var.mod.xgboost]), missing = NA)
        preds <- predict(AISMasterModelo$modelo, newdata = muestra.matrix, ntreelimit = AISMasterModelo$modelo$niter, outputmargin = FALSE)
        cat("Predicciones generadas correctamente\\n")
    }}, error = function(e) {{
        cat("Error al generar predicciones:", e$message, "\\n")
        stop("No se pudieron generar las predicciones")
    }})
    
    # Puntuaciones parciales (contribuciones)
    tryCatch({{
        contrib <- predict(AISMasterModelo$modelo, newdata = muestra.matrix, ntreelimit = AISMasterModelo$modelo$niter, outputmargin = FALSE, predcontrib = TRUE)
        contrib <- as.data.frame(contrib)
        names(contrib) <- paste0("pt_", names(contrib))
        cat("Contribuciones calculadas correctamente\\n")
    }}, error = function(e) {{
        cat("Error al calcular contribuciones, continuando sin ellas:", e$message, "\\n")
        contrib <- data.frame()
    }})
    
    # Combinar resultados
    df$preds <- preds
    if (ncol(contrib) > 0) {{
        df <- cbind(df, contrib)
    }}
    
    # Guardar resultado completo
    tryCatch({{
        write.csv(df, file = "{resultado_completo_path_r}", row.names = FALSE)
        cat("El resultado completo se ha guardado en:", "{resultado_completo_path_r}", "\\n")
    }}, error = function(e) {{
        cat("Error al guardar el archivo completo:", e$message, "\\n")
    }})
    
    # Verificar si existen las columnas bimboId y blmId
    has_bimboId <- "bimboId" %in% colnames(df)
    has_blmId <- "blmId" %in% colnames(df)
    
    # Preparar resultado final
    if (has_bimboId && has_blmId) {{
        resultado_final <- df %>%
            dplyr::select(bimboId, blmId, preds)
    }} else if (has_bimboId) {{
        resultado_final <- df %>%
            dplyr::select(bimboId, preds)
    }} else if (has_blmId) {{
        resultado_final <- df %>%
            dplyr::select(blmId, preds)
    }} else {{
        # Si no existen ninguna de las dos columnas, devolver solo las predicciones
        resultado_final <- data.frame(preds = preds)
    }}
    
    # Guardar resultado final
    tryCatch({{
        write.csv(resultado_final, file = "{resultado_final_path_r}", row.names = FALSE)
        cat("El resultado final se ha guardado en:", "{resultado_final_path_r}", "\\n")
    }}, error = function(e) {{
        cat("Error al guardar el archivo de resultados finales:", e$message, "\\n")
        stop("No se pudo guardar el resultado final")
    }})
    
    cat("Ejecución del script R completada con éxito\\n")
    """
    
    # Escribir el script R en un archivo temporal
    with tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w+') as temp_r_script:
        r_script_path = temp_r_script.name
        temp_r_script.write(r_script_content)
    
    try:
        # Verificar que R está instalado
        try:
            # Intentar ejecutar R con una versión muy simple para verificar disponibilidad
            r_check = subprocess.run(['R', '--version'], 
                                    capture_output=True,
                                    text=True,
                                    timeout=10)
            print("R está instalado en el sistema:")
            print(r_check.stdout.splitlines()[0] if r_check.stdout else "")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("R no está instalado o no es accesible en el PATH")
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
        
        print("Instalando paquetes R necesarios...")
        pkg_install = subprocess.run(['Rscript', pkg_script_path], 
                                    capture_output=True, 
                                    text=True,
                                    timeout=300)  # 5 minutos para instalar paquetes
        
        if pkg_install.returncode != 0:
            print("Advertencia: Posible problema al instalar paquetes R:")
            print(pkg_install.stderr)

        # Ejecutar el script R principal
        print(f"Ejecutando script R: {r_script_path}")
        
        # Usar subprocess.Popen para capturar la salida en tiempo real
        process = subprocess.Popen(['Rscript', '--vanilla', r_script_path], 
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True,
                                  bufsize=1)
        
        # Capturar la salida en tiempo real
        stdout_lines = []
        stderr_lines = []
        
        print("SALIDA DE R:")
        for line in process.stdout:
            print(line.strip())
            stdout_lines.append(line)
            
        # Esperar a que termine el proceso y obtener el código de retorno
        return_code = process.wait()
        
        # Capturar cualquier error pendiente
        for line in process.stderr:
            stderr_lines.append(line)
        
        # Combinar las líneas en un solo string
        stdout_output = ''.join(stdout_lines)
        stderr_output = ''.join(stderr_lines)
        
        if stderr_output:
            print("STDERR DE R:")
            print(stderr_output)

        # Verificar si el proceso terminó con éxito
        if return_code != 0:
            error_msg = f"R script failed with return code {return_code}"
            print(f"❌ {error_msg}")
            if stderr_output:
                error_msg += f"\nSTDERR: {stderr_output}"
            if stdout_output:
                error_msg += f"\nSTDOUT: {stdout_output}"
            raise RuntimeError(error_msg)
        
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
        # Limpieza de archivos temporales
        for file_path in [input_csv_path, r_script_path, pkg_script_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"No se pudo eliminar el archivo temporal {file_path}: {e}")