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
    # configurar opciones de r para evitar problemas comunes
    options(warn = 1)  # mostrar advertencias cuando ocurran

    # crear carpeta de librerías de usuario si no existe
    if (!dir.exists(sys.getenv("r_libs_user"))) {{
        dir.create(sys.getenv("r_libs_user"), recursive = TRUE)
        cat("carpeta r_libs_user creada:", sys.getenv("r_libs_user"), "\\n")
    }}

    .libpaths(c(sys.getenv("r_libs_user"), .libpaths()))

    # paquetes requeridos
    packages_needed <- c("matrix", "xgboost", "dplyr", "fastdummies", "remotes")
    packages_to_install <- packages_needed[!packages_needed %in% installed.packages()[,"package"]]

    # instalar paquetes desde cran o github
    if (length(packages_to_install) > 0) {{
        cat("instalando paquetes necesarios:", paste(packages_to_install, collapse=", "), "\\n")
        for (pkg in packages_to_install) {{
            tryCatch({{
                if (pkg == "remotes") {{
                    install.packages("remotes", repos = "https://cloud.r-project.org", lib = sys.getenv("r_libs_user"), quiet = TRUE)
                }} else if (pkg == "xgboost") {{
                    # verificar que remotes esté instalado antes de instalar xgboost
                    if (!require("remotes", quietly = TRUE)) {{
                        stop("El paquete 'remotes' no está instalado.")
                    }}
                    remotes::install_github("dmlc/xgboost", subdir = "r-package", lib = sys.getenv("r_libs_user"), upgrade = "never")
                }} else {{
                    install.packages(pkg, repos = "https://cloud.r-project.org", lib = sys.getenv("r_libs_user"), quiet = TRUE)
                }}
            }}, error = function(e) {{
                cat("❌ error al instalar", pkg, ":", e$message, "\\n")
            }})
        }}
    }}

    # intentar cargar los paquetes
    for (pkg in packages_needed) {{
        cat("cargando paquete:", pkg, "\\n")
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
            cat("❌ error al cargar el paquete:", pkg, "\\n")
            stop(paste("no se pudo cargar el paquete", pkg))
        }}
    }}
    

    # debugging - mostrar directorio de trabajo actual y modelo a cargar
    cat("directorio de trabajo:", getwd(), "\\n")
    cat("ruta del modelo a cargar:", "{nombre_modelo_r}", "\\n")
    cat("¿el archivo existe?:", file.exists("{nombre_modelo_r}"), "\\n")

    if (!file.exists("{nombre_modelo_r}")) {{
        stop(paste0("el archivo del modelo no existe en la ruta: ", "{nombre_modelo_r}"))
    }}

    tryCatch({{
        df_python <- read.csv("{input_csv_path_r}")
        cat("datos cargados exitosamente, dimensiones:", dim(df_python)[1], "x", dim(df_python)[2], "\\n")
    }}, error = function(e) {{
        cat("error al leer el archivo csv:", e$message, "\\n")
        cat("ruta del archivo:", "{input_csv_path_r}", "\\n")
        stop("no se pudo leer el archivo de entrada")
    }})

    columnas_requeridas <- c("PE_TC_PE_Ventas_AG", "PE_TC_PE_MUNICIPIO_AGR2", "PE_TC_PE_ENTIDAD_AGR")
    columnas_faltantes <- columnas_requeridas[!columnas_requeridas %in% names(df_python)]
    

    # SECCIÓN CORREGIDA PARA MANEJAR COLUMNAS FALTANTES
    if (length(columnas_faltantes) > 0) {{
        cat("Advertencia: Columnas requeridas faltantes:", paste(columnas_faltantes, collapse=", "), "\\n")
        
        # Crear las columnas que faltan
        # Usar un enfoque más seguro para añadir columnas
        for (col in columnas_faltantes) {{
            # Usar la función normal 'transform' en lugar de acceso por doble corchete
            df_python <- transform(df_python, x = NA)
            names(df_python)[names(df_python) == "x"] <- col
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

    # Comprobar existencia de columnas de forma segura
    has_bimboId <- "bimboId" %in% colnames(df)
    has_blmId <- "blmId" %in% colnames(df)
    
    # Preparar resultado final
    # SECCIÓN CORREGIDA PARA LA CREACIÓN DEL DATAFRAME FINAL
    # Usar un enfoque más robusto sin depender del operador pipeline
    if (has_bimboId && has_blmId) {{
        tryCatch({{
            resultado_final <- data.frame(
                bimboId = df$bimboId,
                blmId = df$blmId,
                preds = df$preds
            )
            cat("Dataframe final creado con bimboId y blmId\\n")
        }}, error = function(e) {{
            cat("Error al crear dataframe con bimboId y blmId:", e$message, "\\n")
            # Alternativa en caso de error
            resultado_final <- data.frame(preds = df$preds)
            if (has_bimboId) resultado_final$bimboId <- df$bimboId
            if (has_blmId) resultado_final$blmId <- df$blmId
        }})
    }} else if (has_bimboId) {{
        tryCatch({{
            resultado_final <- data.frame(
                bimboId = df$bimboId,
                preds = df$preds
            )
            cat("Dataframe final creado con solo bimboId\\n")
        }}, error = function(e) {{
            cat("Error al crear dataframe con bimboId:", e$message, "\\n")
            resultado_final <- data.frame(preds = df$preds)
            resultado_final$bimboId <- df$bimboId
        }})
    }} else if (has_blmId) {{
        tryCatch({{
            resultado_final <- data.frame(
                blmId = df$blmId,
                preds = df$preds
            )
            cat("Dataframe final creado con solo blmId\\n")
        }}, error = function(e) {{
            cat("Error al crear dataframe con blmId:", e$message, "\\n")
            resultado_final <- data.frame(preds = df$preds)
            resultado_final$blmId <- df$blmId
        }})
    }} else {{
        # Si no existen ninguna de las dos columnas, devolver solo las predicciones
        resultado_final <- data.frame(preds = preds)
        resultado_final <- data.frame(preds = df$preds)
        cat("Dataframe final creado solo con predicciones\\n")
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