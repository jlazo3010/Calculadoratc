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
    if (Sys.getenv("R_LIBS_USER") == "") {{
        Sys.setenv(R_LIBS_USER = paste0(Sys.getenv("HOME"), "/R/", R.version$platform, "/library"))
    }}
    if (!dir.exists(Sys.getenv("R_LIBS_USER"))) {{
        dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)
    }}
    .libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

    if (!require("xgboost", quietly = TRUE)) {{
        cat("⏳ Instalando xgboost desde CRAN...\n")
        install.packages("xgboost", repos = "https://cloud.r-project.org", lib = Sys.getenv("R_LIBS_USER"))
        cat("✅ Instalación de xgboost finalizada\n")
    }} else {{
        cat("✅ xgboost ya está instalado\n")
    }}

    for (pkg in c("Matrix", "dplyr", "fastDummies")) {{
        if (!require(pkg, character.only = TRUE)) {{
            install.packages(pkg, repos = "https://cloud.r-project.org", lib = Sys.getenv("R_LIBS_USER"))
        }}
    }}

    cat("Cargando datos desde CSV...\n")
    df_python <- read.csv("{input_csv_path_r}")
    cat("Datos cargados. Dimensiones:", dim(df_python)[1], "x", dim(df_python)[2], "\n")

    columnas_requeridas <- c("PE_TC_PE_Ventas_AG", "PE_TC_PE_MUNICIPIO_AGR2", "PE_TC_PE_ENTIDAD_AGR")
    columnas_faltantes <- columnas_requeridas[!columnas_requeridas %in% names(df_python)]
    if (length(columnas_faltantes) > 0) {{
        for (col in columnas_faltantes) {{
            df_python[[col]] <- NA
        }}
    }}

    columnas_dummy <- columnas_requeridas[columnas_requeridas %in% names(df_python)]
    if (length(columnas_dummy) > 0) {{
        df <- fastDummies::dummy_cols(df_python, select_columns = columnas_dummy, remove_selected_columns = TRUE)
    }} else {{
        df <- df_python
    }}

    cat("Cargando modelo...\n")
    load("{nombre_modelo_r}")
    cat("Modelo cargado\n")

    var.mod.xgboost <- AISMasterModelo$modelo$feature_names
    for (var in var.mod.xgboost) {{
        if (!(var %in% colnames(df))) {{
            df[[var]] <- NA
        }}
    }}

    df <- data.frame(lapply(df, function(x) {{ replace(x, is.nan(x) | is.infinite(x), NA) }}))

    cat("Generando predicciones...\n")
    muestra.matrix <- xgboost::xgb.DMatrix(as.matrix(df[, var.mod.xgboost]), missing = NA)
    preds <- predict(AISMasterModelo$modelo, newdata = muestra.matrix, ntreelimit = AISMasterModelo$modelo$niter)
    df$preds <- preds
    write.csv(df, file = "{resultado_final_path_r}", row.names = FALSE)
    cat("✅ Predicciones generadas y guardadas\n")
    """

    with tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w+') as temp_r_script:
        r_script_path = temp_r_script.name
        temp_r_script.write(r_script_content)

    try:
        print("Iniciando ejecución de R...")
        process = subprocess.Popen(['Rscript', '--vanilla', r_script_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   bufsize=1)

        stdout_lines, stderr_lines = [], []
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            print(f"R: {line.strip()}")
            stdout_lines.append(line.strip())

        for line in iter(process.stderr.readline, ''):
            if not line:
                break
            print(f"R ERROR: {line.strip()}")
            stderr_lines.append(line.strip())

        exit_code = process.wait(timeout=1200)

        if exit_code != 0:
            raise RuntimeError("\n".join(stderr_lines))

        if os.path.exists(resultado_final_path):
            return pd.read_csv(resultado_final_path)
        else:
            raise FileNotFoundError(f"El archivo de resultados no se ha creado en: {resultado_final_path}")

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("Timeout: El script R tardó demasiado tiempo y fue cancelado.")

    finally:
        for file_path in [input_csv_path, r_script_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"No se pudo eliminar el archivo temporal {file_path}: {e}")