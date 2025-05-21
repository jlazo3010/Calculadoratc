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

def ejecutar_modelo_ais(nombre_muestra=None):

    start_time = time.time()

    if nombre_muestra is None:
        raise ValueError("Debe proporcionar un DataFrame como nombre_muestra")

    df = nombre_muestra.copy()

    nombres_originales = df.columns.tolist()
    nombres_limpios = [col.encode('ascii', 'ignore').decode('ascii') for col in nombres_originales]
    nombres_limpios = [re.sub(r'[^0-9a-zA-Z_]', '', col) for col in nombres_limpios]
    rename_dict = dict(zip(nombres_originales, nombres_limpios))
    df = df.rename(columns=rename_dict)

    path = os.getcwd().replace("\\", "/")
    if not path.endswith("/"):
        path += "/"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    nombre_modelo = os.path.join(script_dir, "AISMaster_Modelo_20241223131859.RData")
    nombre_modelo_r = nombre_modelo.replace("\\", "/")

    if not os.path.exists(nombre_modelo):
        raise FileNotFoundError(f"El archivo del modelo no existe en la ruta: {nombre_modelo}")

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as temp_input_file:
        input_csv_path = temp_input_file.name
        df.to_csv(input_csv_path, index=False)

    if not input_csv_path or not os.path.exists(input_csv_path):
        raise FileNotFoundError("No se pudo crear el archivo temporal para los datos")

    input_csv_path_r = input_csv_path.replace("\\", "/")
    resultado_final_path = os.path.join(path, "resultado_final.csv")
    resultado_final_path_r = resultado_final_path.replace("\\", "/")

    r_script_content = f"""
    options(warn = 1)
    
    # Configurar rutas de bibliotecas
    if (Sys.getenv("R_LIBS_USER") == "") {{
        Sys.setenv(R_LIBS_USER = paste0(Sys.getenv("HOME"), "/R/", R.version$platform, "/library"))
    }}
    if (!dir.exists(Sys.getenv("R_LIBS_USER"))) {{
        dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)
    }}
    .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
    
    # Función para limpiar bloqueos de instalación
    clean_installation_locks <- function() {{
        lib_path <- Sys.getenv("R_LIBS_USER")
        lock_dirs <- list.files(lib_path, pattern = "^00LOCK-", full.names = TRUE)
        if (length(lock_dirs) > 0) {{
            cat("🧹 Limpiando directorios de bloqueo:", paste(basename(lock_dirs), collapse = ", "), "\\n")
            for (lock_dir in lock_dirs) {{
                if (dir.exists(lock_dir)) {{
                    unlink(lock_dir, recursive = TRUE, force = TRUE)
                }}
            }}
        }}
    }}
    
    # Función para instalar paquetes con reintentos
    install_package_with_retry <- function(pkg_name, max_attempts = 3) {{
        for (attempt in 1:max_attempts) {{
            cat("🔄 Intento", attempt, "de", max_attempts, "para instalar", pkg_name, "\\n")
            
            # Limpiar bloqueos antes de cada intento
            clean_installation_locks()
            
            # Intentar instalación
            tryCatch({{
                if (!require(pkg_name, character.only = TRUE, quietly = TRUE)) {{
                    install.packages(pkg_name, 
                                   repos = "https://cloud.r-project.org", 
                                   lib = Sys.getenv("R_LIBS_USER"),
                                   dependencies = TRUE,
                                   type = "both")
                    
                    # Verificar si la instalación fue exitosa
                    if (require(pkg_name, character.only = TRUE, quietly = TRUE)) {{
                        cat("✅", pkg_name, "instalado exitosamente\\n")
                        return(TRUE)
                    }}
                }}
                else {{
                    cat("✅", pkg_name, "ya está disponible\\n")
                    return(TRUE)
                }}
            }}, error = function(e) {{
                cat("❌ Error en intento", attempt, ":", e$message, "\\n")
                if (attempt < max_attempts) {{
                    cat("⏳ Esperando 5 segundos antes del siguiente intento...\\n")
                    Sys.sleep(5)
                }}
            }})
        }}
        
        cat("❌ No se pudo instalar", pkg_name, "después de", max_attempts, "intentos\\n")
        return(FALSE)
    }}
    
    # Instalar paquetes requeridos
    required_packages <- c("Matrix", "dplyr", "fastDummies", "xgboost")
    
    for (pkg in required_packages) {{
        success <- install_package_with_retry(pkg)
        if (!success) {{
            cat("❌ CRÍTICO: No se pudo instalar", pkg, "\\n")
            quit(status = 1)
        }}
    }}
    
    # Cargar bibliotecas
    cat("📚 Cargando bibliotecas...\\n")
    library(Matrix)
    library(dplyr)
    library(fastDummies)
    library(xgboost)
    
    cat("📊 Cargando datos desde CSV...\\n")
    tryCatch({{
        df_python <- read.csv("{input_csv_path_r}")
        cat("✅ Datos cargados. Dimensiones:", dim(df_python)[1], "x", dim(df_python)[2], "\\n")
    }}, error = function(e) {{
        cat("❌ Error al cargar datos:", e$message, "\\n")
        quit(status = 1)
    }})

    # Verificar y agregar columnas requeridas
    columnas_requeridas <- c("PE_TC_PE_Ventas_AG", "PE_TC_PE_MUNICIPIO_AGR2", "PE_TC_PE_ENTIDAD_AGR")
    columnas_faltantes <- columnas_requeridas[!columnas_requeridas %in% names(df_python)]
    
    if (length(columnas_faltantes) > 0) {{
        cat("⚠️  Agregando columnas faltantes:", paste(columnas_faltantes, collapse = ", "), "\\n")
        for (col in columnas_faltantes) {{
            df_python[[col]] <- NA
        }}
    }}

    # Crear variables dummy
    columnas_dummy <- columnas_requeridas[columnas_requeridas %in% names(df_python)]
    if (length(columnas_dummy) > 0) {{
        cat("🔄 Creando variables dummy para:", paste(columnas_dummy, collapse = ", "), "\\n")
        df <- fastDummies::dummy_cols(df_python, select_columns = columnas_dummy, remove_selected_columns = TRUE)
    }} else {{
        df <- df_python
    }}

    # Cargar modelo
    cat("🤖 Cargando modelo desde:", "{nombre_modelo_r}", "\\n")
    tryCatch({{
        load("{nombre_modelo_r}")
        cat("✅ Modelo cargado exitosamente\\n")
    }}, error = function(e) {{
        cat("❌ Error al cargar modelo:", e$message, "\\n")
        quit(status = 1)
    }})

    # Verificar variables del modelo
    if (!exists("AISMasterModelo")) {{
        cat("❌ Error: No se encontró el objeto AISMasterModelo en el archivo cargado\\n")
        quit(status = 1)
    }}
    
    var.mod.xgboost <- AISMasterModelo$modelo$feature_names
    cat("📋 Variables requeridas por el modelo:", length(var.mod.xgboost), "\\n")
    
    # Agregar variables faltantes para el modelo
    for (var in var.mod.xgboost) {{
        if (!(var %in% colnames(df))) {{
            df[[var]] <- NA
        }}
    }}

    # Limpiar datos
    cat("🧹 Limpiando datos (NaN e infinitos)...\\n")
    df <- data.frame(lapply(df, function(x) {{ replace(x, is.nan(x) | is.infinite(x), NA) }}))

    # Generar predicciones
    cat("🔮 Generando predicciones...\\n")
    tryCatch({{
        muestra.matrix <- xgboost::xgb.DMatrix(as.matrix(df[, var.mod.xgboost]), missing = NA)
        preds <- predict(AISMasterModelo$modelo, newdata = muestra.matrix, ntreelimit = AISMasterModelo$modelo$niter)
        df$preds <- preds
        
        # Guardar resultados
        write.csv(df, file = "{resultado_final_path_r}", row.names = FALSE)
        cat("✅ Predicciones generadas y guardadas en:", "{resultado_final_path_r}", "\\n")
        cat("📊 Predicciones estadísticas - Min:", min(preds), "Max:", max(preds), "Media:", mean(preds), "\\n")
        
    }}, error = function(e) {{
        cat("❌ Error durante predicciones:", e$message, "\\n")
        quit(status = 1)
    }})
    
    cat("🎉 Proceso completado exitosamente\\n")
    """

    with tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w+') as temp_r_script:
        r_script_path = temp_r_script.name
        temp_r_script.write(r_script_content)

    try:
        print("🚀 Iniciando ejecución de R...")
        process = subprocess.Popen(['Rscript', '--vanilla', r_script_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1,
                                   universal_newlines=True)

        stdout_lines, stderr_lines = [], []
        
        # Leer output en tiempo real
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(f"R: {line}")
                stdout_lines.append(line)

        # Leer errores
        stderr_output = process.stderr.read()
        if stderr_output:
            stderr_lines = stderr_output.strip().split('\n')
            for line in stderr_lines:
                if line.strip():
                    print(f"R ERROR: {line.strip()}")

        exit_code = process.wait(timeout=1800)  # Aumentar timeout a 30 minutos

        if exit_code != 0:
            error_msg = f"El script R terminó con código de error {exit_code}"
            if stderr_lines:
                error_msg += ":\n" + "\n".join(stderr_lines)
            raise RuntimeError(error_msg)

        # Verificar que el archivo de resultados existe
        if os.path.exists(resultado_final_path):
            print("✅ Archivo de resultados encontrado, cargando...")
            resultado_df = pd.read_csv(resultado_final_path)
            print(f"📊 Resultados cargados: {len(resultado_df)} filas, {len(resultado_df.columns)} columnas")
            
            # Limpiar archivo de resultados
            try:
                os.unlink(resultado_final_path)
            except:
                pass
                
            return resultado_df
        else:
            raise FileNotFoundError(f"El archivo de resultados no se ha creado en: {resultado_final_path}")

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("⏰ Timeout: El script R tardó demasiado tiempo y fue cancelado.")
    
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        raise

    finally:
        # Limpiar archivos temporales
        for file_path in [input_csv_path, r_script_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"⚠️  No se pudo eliminar el archivo temporal {file_path}: {e}")

        elapsed_time = time.time() - start_time
        print(f"⏱️  Tiempo total de ejecución: {elapsed_time:.2f} segundos")