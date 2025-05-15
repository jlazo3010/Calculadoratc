import streamlit as st
import re
import pandas as pd
import boto3
import io
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from modelo_ais import ejecutar_modelo_ais
import numpy as np
import subprocess
import statsmodels.api as sm
import joblib 
import os
import time
import pandas as pd
import numpy as np
import warnings
import subprocess
import tempfile
import re
import sys
warnings.filterwarnings('ignore')
import math
import pickle 
import json

st.set_page_config(page_title="App de Clientes", layout="centered")
# ---------------- CONFIGURACIÓN ----------------

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
BUCKET_NAME = os.environ['S3_BUCKET']
ARCHIVO_S3 = 'tconetcacalculadora.csv'
BASEID = 'Base_martin.csv'
CPID = 'CPID.csv'
AIS = 'base_AIS.csv'
USU = 'Usuarios.csv'

# Verificar que las variables se cargaron correctamente
if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME]):
    print("⚠️ Las variables de entorno no se cargaron correctamente.")
else:
    print("✅ Las variables de entorno se cargaron correctamente.")

# Carga del cliente S3
s3 = boto3.client('s3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# ---------------- FUNCIONES ----------------

def BimboIDbase():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=BASEID)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'blmId': str})
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=['blmId', 'Morosidad_Promedio', 'Gradient Boosting_Proba',
                                       'Decil_ventas','PromedioVisitasXMesBimbo','ventaPromedioSemanalUlt12Semanas',
                                       'Giro_de_Cliente','MontoMinCredito','DiasConCreditoVigente'])
        else:
            raise e
    return df

# Cargar la tabla BIMBOID
tabla_bimbo = BimboIDbase()
tabla_bimbo["blmId"] = tabla_bimbo["blmId"].astype(str)

def CPbase():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=CPID)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'d_codigo': str})
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=['d_codigo', 'd_asenta', 'd_tipo_asenta', 'D_mnpio', 'd_estado','d_zona'])
        else:
            raise e
    return df

# Cargar la tabla MUNICIPIOS
tabla_CPID = CPbase()
tabla_CPID["d_codigo"] = tabla_CPID["d_codigo"].astype(str)

def AISbase():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=AIS)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'bimboId': str}, encoding='latin1')
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=[])
        else:
            raise e
    return df

# Cargar la tabla MUNICIPIOS
tabla_AIS = AISbase()
tabla_AIS["bimboId"] = tabla_AIS["bimboId"].astype(str)
tabla_AIS.rename(columns={tabla_AIS.columns[0]: 'PE_TC_PE_MUNICIPIO_C'}, inplace=True)

##Base de usuarios
def USUARIOS():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=USU)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'Usuario': str, 'Pass' : str})
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=[])
        else:
            raise e
    return df

# Cargar la tabla MUNICIPIOS
tabla_USU = USUARIOS()
tabla_USU["Usuario"] = tabla_USU["Usuario"].astype(str)
tabla_USU["Pass"] = tabla_USU["Pass"].astype(str)


def cargar_base():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=ARCHIVO_S3)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'nombre': str, 'BimboID': str,
                                                                     'blmId' : str, 'Solicitud':str,
                                                                     'Usuario_registro':str})
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=['nombre', 'edad', 'genero', 'BimboID'])
        else:
            raise e
    return df

def guardar_base(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=ARCHIVO_S3, Body=buffer.getvalue())

def eliminar_registro(Solicitud):
    df = cargar_base()
    df = df[df['Solicitud'] != Solicitud]
    guardar_base(df)

# Tabla de reglas
reglas = pd.DataFrame({
    "ScoreFico_min": [0, 0, 549, 549, 649, 0, 606],
    "ScoreFico_max": [548, 548, 648, 648, 1000, 605, 1000],
    "MicroScore_min": [-10, 621, -10, 621, -10, 661, 661],
    "MicroScore_max": [620, 660, 620, 660, 660, 1000, 1000],
    "Grupo": [1, 2, 3, 4, 5, 6, 7]
})

### Decil de riesgos
def asignar_decil(score_fico, micro_score):
    for _, regla in reglas.iterrows():
        if (regla["ScoreFico_min"] <= score_fico <= regla["ScoreFico_max"] and
            regla["MicroScore_min"] <= micro_score <= regla["MicroScore_max"]):
            return regla["Grupo"]
    return None  # Si no entra en ningún rango


## Decil RL
def obtener_decil(valor):
    rangos = [
        (0, 0.129, 10),
        (0.13, 0.216, 9),
        (0.217, 0.267, 8),
        (0.268, 0.31, 7),
        (0.311, 0.356, 6),
        (0.357, 0.405, 5),
        (0.406, 0.461, 4),
        (0.462, 0.523, 3),
        (0.524, 0.599, 2),
        (0.6, 1, 1),
    ]
    
    for minimo, maximo, decil in rangos:
        if minimo <= valor <= maximo:
            return decil
    return None  # Si no cae en ningún rango

## Decil AIS
def obtener_decil_AIS(valor):
    rangos = [
        (0, 0.315341234, 10),
        (0.315347373, 0.358658999, 9),
        (0.358679146, 0.390403807, 8),
        (0.390479475, 0.422031224, 7),
        (0.422036052, 0.455979526, 6),
        (0.456012547, 0.503053308, 5),
        (0.503120899, 0.544607878, 4),
        (0.544611394, 0.582605958, 3),
        (0.582607388, 0.626625597, 2),
        (0.626704872, 1, 1),
    ]
    
    for minimo, maximo, decil in rangos:
        if minimo <= valor <= maximo:
            return decil
    return None  # Si no cae en ningún rango

def asignar_desiscion(grupo, Microscore):
    if grupo == 6 or (-7 <= Microscore <= -1):
        return "Rechazado"
    else:
        return "Aceptado"

############################## FUNCIONES PARA MODELO RL ###############################
# datos numéricos
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
 
# datos categóricos con muchas categorías
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown='error'):
        self.encoders = None
        self.handle_unknown = handle_unknown
    
    def fit(self, X, y=None):
        self.encoders = [LabelEncoder() for _ in range(X.shape[1])]
        for i, encoder in enumerate(self.encoders):
            encoder.fit(X[:, i])
        return self
    
    def transform(self, X):
        X_encoded = np.zeros_like(X, dtype=int)
        for i, encoder in enumerate(self.encoders):
            try:
                X_encoded[:, i] = encoder.transform(X[:, i])
            except ValueError:
                if self.handle_unknown == 'ignore':
                    X_encoded[:, i] = -1  # Valor para categorías desconocidas
                else:
                    raise
        return X_encoded
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
 
categorical_transformer_label = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('label', MultiColumnLabelEncoder(handle_unknown='ignore'))
])

# Función para cargar el modelo
def cargar_modelo(ruta_modelo='modelo_regresion_logistica_v2.pkl'):
    """
    Carga el modelo de regresión logística guardado previamente con manejo de errores
    """
    try:
        with open(ruta_modelo, 'rb') as file:
            modelo_cargado = pickle.load(file)
        return modelo_cargado
    except AttributeError:
        st.error("Error al cargar el modelo: La clase MultiColumnLabelEncoder no fue encontrada. Verifica que esté definida correctamente.")
        # Define una respuesta predeterminada o alternativa
        return None
 
# Función para preprocesar nuevos datos
def preprocesar_nuevos_datos(datos_nuevos, modelo_cargado):
    """
    Preprocesa nuevos datos utilizando el preprocesador guardado
    """
    # Aplicar el mismo preprocesamiento que se utilizó en el entrenamiento
    preprocessor = modelo_cargado['preprocessor'] # procesador de datos (se requieren las funciones de pipe line de cada tipo de dato)
    X_prepared = preprocessor.transform(datos_nuevos)
   
    # Convertir a DataFrame con los nombres de las características
    X_df = pd.DataFrame(X_prepared, columns=modelo_cargado['feature_names'])
   
    # Seleccionar solo las características que se utilizaron en el modelo final
    X_selected = X_df[modelo_cargado['selected_features']]
   
    # Eliminar columnas altamente correlacionadas
    X_selected = X_selected.drop(columns=modelo_cargado['drop_columns'], errors='ignore')
   
    return X_selected
 
# Función para predecir probabilidades
def predecir_probabilidades(datos_preprocesados, modelo_cargado):
    """
    Predice las probabilidades utilizando el modelo cargado
    """
    # Añadir constante para el intercepto
    X_const = sm.add_constant(datos_preprocesados)
   
    # Predecir probabilidades
    probabilidades = modelo_cargado['model'].predict(X_const)
   
    return probabilidades
 
# Cargar el modelo
modelo_cargado = cargar_modelo('modelo_regresion_logistica_v2.pkl')


############################## FUNCIONES PARA ASIGNACIÓN FINAL DE DECIL ###############################
# Inicializar la matriz de clusters con -1 (sin asignar)
matriz_clusters = np.full((10, 10), -1, dtype=int)

# Grupo 1
grupo_1_coords = [
    (0, 0), (0, 1), (0, 2),  # RL-1 con XGB-1,2,3
    (1, 0), (1, 1), (1, 2),   # RL-2 con XGB-1,2,3
    (2, 0), (2, 1), (2, 2)
]

# Grupo 2
grupo_2_coords = [
(0,3),	(0,4),
(1,3),	(1,4),
(2,3),	(2,4),
(3,0),	(3,1),	
(3,2),   (3,3),
(3,4),   (4,0),
(4,1),	(4,2),
(4,3),	(4,4)
]

# Grupo 3
grupo_3_coords = [
(0,5),	(0,6),
(1,5),	(1,6),
(2,5),	(2,6),
(3,5),	(3,6),
(4,5),	(4,6),
(5,0),	(5,1),	
(5,2),	(5,3),	
(5,4),  (5,5),
(5,6),   (6,0),	
(6,1),	(6,2)
]

# Grupo 4
grupo_4_coords = [
    (3,7),	(3,8),	(3,9),
    (4,7),	(4,8),	(4,9),
    (5,7),	(5,8),	(5,9)
]

# Grupo 5
grupo_5_coords = [
    (6,3),	(6,4),	(6,5),	
    (6,6),	(6,7),	(6,8),
    (7,3),	(7,4),	(7,5),	
    (7,6),  (8,3)
    ]

# Grupo 6
grupo_6_coords = [
    (6,9),
    (7,7),	(7,8),	(7,9),
    (8,4),	(8,5),	(8,6),	(8,7),	(8,8),	(8,9),
    (9,3),	(9,4),	(9,5),	(9,6),	(9,7),	(9,8),	(9,9)
]

# Grupo 7
grupo_7_coords = [
    (7,0),	(7,1),	(7,2),
    (8,0),	(8,1),	(8,2),
    (9,0),	(9,1),	(9,2)
]

# Grupo 8
grupo_8_coords = [
    (0,7),	(0,8),	(0,9),
    (1,7),	(1,8),	(1,9),
    (2,7),	(2,8),	(2,9)
]


# Asignar grupos a la matriz
for i, j in grupo_1_coords:
    matriz_clusters[i, j] = 0  # Grupo 1 (índice 0)

for i, j in grupo_2_coords:
    matriz_clusters[i, j] = 1  # Grupo 2 (índice 1)

for i, j in grupo_3_coords:
    matriz_clusters[i, j] = 2  # Grupo 3 (índice 2)

for i, j in grupo_4_coords:
    matriz_clusters[i, j] = 3  # Grupo 4 (índice 3)

for i, j in grupo_5_coords:
    matriz_clusters[i, j] = 4  # Grupo 5 (índice 4)

for i, j in grupo_6_coords:
    matriz_clusters[i, j] = 5  # Grupo 6 (índice 5)

for i, j in grupo_7_coords:
    matriz_clusters[i, j] = 6  # Grupo 7 (índice 6)

for i, j in grupo_8_coords:
    matriz_clusters[i, j] = 7  # Grupo 8 (índice 7)

# 11. Función para calificar un solo crédito
deciles_rl = [0.111872366035592, 0.1936463963737612, 0.2503706582687602, 0.2917784224934482, 0.3346987520198585, 0.3817219439752034, 0.4370613082398364, 0.4983681906481146, 0.578388701340094]
deciles_xgb = [0.3153418481349942, 0.3586630284786228, 0.39042650759220165, 0.4220331549644468, 0.455979526042938, 0.503093862533569, 0.5446103394031524, 0.5826071023941041, 0.626696944236755]
def calificar_credito(prob_rl, prob_xgb, deciles_rl, deciles_xgb):
    """Califica un solo crédito según su probabilidad RL y XGB."""
    # Asignar decil RL
    decil_rl = 1  # Valor predeterminado
    for i, umbral in enumerate(deciles_rl):
        if prob_rl >= umbral:
            decil_rl = i + 2
    
    # Asignar decil XGB
    decil_xgb = 1  # Valor predeterminado
    for i, umbral in enumerate(deciles_xgb):
        if prob_xgb >= umbral:
            decil_xgb = i + 2
    
    # Obtener grupo
    grupo_num = -1  # Sin asignar por defecto
    if 0 <= decil_rl-1 < 10 and 0 <= decil_xgb-1 < 10:
        grupo_num = int(matriz_clusters[decil_rl-1, decil_xgb-1])
    
    # Mapeo de números de grupo a nombres
    nombres_grupos = {
        0: 'Grupo 1 - Riesgo Muy Bajo',
        1: 'Grupo 2 - Riesgo Bajo',
        2: 'Grupo 3 - Riesgo Medio-Bajo',
        3: 'Grupo 4 - Riesgo Medio',
        4: 'Grupo 5 - Riesgo Medio-Alto',
        5: 'Grupo 6 - Riesgo Alto',
        6: 'Grupo 7 - Alta Discrepancia XGB',
        7: 'Grupo 8 - Alta Discrepancia RL',
    }
    
    if grupo_num >= 0:
        nombre_grupo = nombres_grupos.get(grupo_num, f'Grupo {grupo_num+1}')
    else:
        nombre_grupo = 'Sin Grupo Asignado'

    grupo_num2 = grupo_num + 1
    
    return nombre_grupo, grupo_num2

def montos_grupo(numerodegrupo, Desiscion):
    rangos = [
        (1, 37500, 50271),
        (2, 28066, 37499),
        (3, 19585, 28065),
        (4, 13818, 19584),
        (5, 10482, 13817),
        (6, 4000, 10482),
        (7, 23667, 45939),
        (8, 10816, 23883)
    ]

    if Desiscion == "Aceptado":
        for grupo, minimo, maximo in rangos:
            if numerodegrupo == grupo:
                return math.ceil(minimo), math.ceil(maximo)
    return 0, 0  # Si no se encuentra el grupo o la decisión no es "Aceptado"

def oferta_final(min_oferta, max_oferta, oferta_original):
    if min_oferta <= oferta_original <= max_oferta:
        return oferta_original
    elif oferta_original < min_oferta:
        return min_oferta
    elif max_oferta < oferta_original:
        return max_oferta

################################# Manejo de la limpieza del formulario
if 'limpiar_formulario' in st.session_state and st.session_state['limpiar_formulario']:
    # Restablecer valores predeterminados para todos los campos
    st.session_state['Solicitud'] = 0
    st.session_state['nombre'] = ""
    st.session_state['comentarios'] = ""
    st.session_state['Oferta'] = 0
    st.session_state['nombre'] = ""
    st.session_state['edad'] = 18
    st.session_state['LLAMADA'] = ""
    st.session_state['CP'] = ""
    st.session_state['genero'] = "Masculino"
    st.session_state['Dependientes'] = "0"
    if 'tipo_negocio_especificado' not in st.session_state:
        st.session_state['tipo_negocio_especificado'] = ""
    st.session_state['Edo_civil'] = "Casado"
    st.session_state['Tipo_negocio'] = "ABARROTES"
    st.session_state['BimboID'] = ""
    st.session_state['blmId'] = ""
    
    # Gestionar MicroScore y ScoreFico según InfoCre
    st.session_state['InfoCre'] = "Si"  # Valor predeterminado
    st.session_state['MicroScore'] = 0
    st.session_state['ScoreFico'] = 0
    
    # Quitar la bandera para evitar limpiar en cada recarga
    st.session_state['limpiar_formulario'] = False

    # Quitar checkboxes
    st.session_state["INE"] = False
    st.session_state["Domicilio"] = False
    st.session_state["CURP"] = False
    st.session_state["RFC"] = False
    st.session_state["SPEI"] = False
    st.session_state["FOTO"] = False

# ---------------- INTERFAZ ----------------

st.markdown("""
    <style>
    .stButton>button {
        background-color: white;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
            
    /* Cambiar el fondo del contenedor principal del formulario */
    div[data-testid="stForm"] {
        background-color: #002b54; /* Color de fondo */
        padding: 20px;              /* Padding alrededor del formulario */
        border-radius: 10px;        /* Esquinas redondeadas */
    }

    h1, h2, h3, h4, h5, h6, h7 {
        color: white !important;
    }
            
    /* Personalizar los campos del formulario */
    .stTextInput, .stNumberInput, .stDateInput, .stSelectbox, .stCheckbox{
        background-color: #0056b1; /* Fondo blanco para los inputs */
        border: 2px solid #ffffff; /* Borde gris */
        border-radius: 8px;        /* Esquinas redondeadas */
        padding: 10px;              /* Padding interno */
        padding-bottom: 20px;
    }
            
    /* Personalizar los campos del formulario */
    .stCheckbox{
        background-color: #47a1ff; /* Fondo blanco para los inputs */
        border: 2px solid #ffffff; /* Borde gris */
        border-radius: 8px;        /* Esquinas redondeadas */
        padding: 10px;              /* Padding interno */
        padding-bottom: 20px;
    }
            
    /* Cambiar color de los labels de los campos a blanco */
    label, .stCheckbox > div {
        color: white !important;
        font-weight: bold;
    }

    .stMarkdown p {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([8, 3])  # Ajusta proporción a tu gusto

with col1:
    st.title("Calculadora T-Conecta")

with col2:
    st.image("imagen.png", width=150)

st.markdown("---")

if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

if 'usuario_actual' not in st.session_state:
    st.session_state['usuario_actual'] = ""

# Pantalla de inicio de sesión
if not st.session_state['autenticado']:
    st.title("🔒 Inicio de Sesión")
    
    with st.form("login_form"):
        usuario = st.text_input("Usuario")
        contraseña = st.text_input("Contraseña", type="password")
        submit_login = st.form_submit_button("Iniciar Sesión")
        
        if submit_login:
            # Cargar base de usuarios
            usuarios_df = tabla_USU
            
            # Verificar credenciales
            if not usuarios_df.empty and ((usuarios_df['Usuario'] == usuario) & (usuarios_df['Pass'] == contraseña)).any():
                st.session_state['autenticado'] = True
                st.session_state['usuario_actual'] = usuario
                st.success("✅ Inicio de sesión exitoso!")
                st.rerun()  # Recargar la página para mostrar el formulario
            else:
                st.error("❌ Usuario o contraseña incorrectos.")
else:
    # Aquí iría todo tu código actual del formulario
    st.title(f"📋 Registro de Clientes - Bienvenido {st.session_state['usuario_actual']}")
    with st.form("form_cliente"):
        col1, col2 = st.columns(2)
        with col1:
            Solicitud = st.number_input("Solicitud", min_value=0, max_value=100, key = "Solicitud")

            nombre = st.text_input("Nombre completo", max_chars=50, key = "nombre",value="")
            if nombre and len(nombre) < 3:
                st.warning("El nombre debe tener al menos 3 caracteres.")

            Edad = st.number_input("Edad", min_value=18, max_value=100, step=1,key = "edad")

            InfoCre = st.selectbox("¿Cuenta con información de Crédito?", ["Si", "No"], key="InfoCre")

            MicroScore = st.number_input(
                "MicroScore",
                min_value=-7,
                max_value=1000,
                step=1,
                key="MicroScore"
            )

            ScoreFico = st.number_input("ScoreFico", 
                min_value=0, 
                max_value=1000, 
                step=1, 
                key="ScoreFico"
            )

            Edo_civil = st.selectbox("Estado civil", ["Casado", "Union libre", "Soltero", "Viudo","Divorciado"],key = "Edo_civil")


            Oferta = st.number_input("Oferta", min_value=0, max_value=10000000, step=1,key = "Oferta")

            Comentarios = st.text_input("Comentario", max_chars=150, key = "comentarios",value="")

            llamada = st.selectbox("¿Se realizo llamada?", ["Si", "No"],key = "LLAMADA")

        with col2:
            genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"],key = "genero")

            Dependientes = st.selectbox("Dependiente", ["0", "1", "2", "3", "4", "5", "+5"],key = "Dependientes")

            CP = st.text_input("Código postal", max_chars=50,key = "CP",value="")
            resultadosCP = pd.DataFrame()

            if CP:
                if len(CP) != 5:
                    st.warning("El CP debe tener al menos 5 caracteres.")
                else:
                    resultadosCP = tabla_CPID[tabla_CPID["d_codigo"] == CP]
                    if not resultadosCP.empty:
                        st.success("✅ Datos encontrados en BaseCP")
                    else:
                        st.error("No se encontraron datos para ese CP.")

            Tipo_negocio = st.selectbox("Tipo de negocio", 
                            ['ABARROTES','CONSUMOS','MISCELANEAS','OTROS','SIN INFORMACION'],
                            key="Tipo_negocio",
                            on_change=None)  # Asegura que se actualice la interfaz

            if st.session_state.Tipo_negocio == "SIN INFORMACION":
                tipo_negocio_especificado = st.text_input("Especifique el tipo de negocio:", 
                                                        key="tipo_negocio_especificado")
            
            BimboID = st.text_input("BimboID", max_chars=15,key = "BimboID",value="")
            tabla_AIS["bimboId"] = tabla_AIS["bimboId"].astype(str)
            resultadosAIS = pd.DataFrame()

            if BimboID:
                if len(BimboID) < 3:
                    st.warning("El BimboID debe tener al menos 3 caracteres.")
                else:
                    resultadosAIS = tabla_AIS[tabla_AIS["bimboId"] == BimboID]
                    if not resultadosAIS.empty:
                        st.success("✅ Datos encontrados en AIS")
                        resultadosAIS
                    else:
                        st.error("No se encontraron datos para ese BimboID.")

                        tabla_AIS["bimboId"] = tabla_AIS["bimboId"].astype(str)


            blmId = st.text_input("blmId", max_chars=15,key = "blmId",value="")
            tabla_bimbo["blmId"] = tabla_bimbo["blmId"].astype(str)
            resultados = pd.DataFrame()

            if blmId:
                if len(blmId) < 3:
                    st.warning("El blmId debe tener al menos 3 caracteres.")
                else:
                    resultados = tabla_bimbo[tabla_bimbo["blmId"] == blmId]
                    if not resultados.empty:
                        st.success("✅ Datos encontrados en Base Bimbo")
                    else:
                        st.error("No se encontraron datos para ese blmId.")

                        tabla_bimbo["blmId"] = tabla_bimbo["blmId"].astype(str)

            valido_ine = st.checkbox("Validación de INE" , key="INE")
            valido_domicilio = st.checkbox("Validación de Domicilio", key="Domicilio")
            valido_curp = st.checkbox("Validación de CURP", key="CURP")
            valido_rfc = st.checkbox("Validación de RFC", key="RFC")
            valido_spei = st.checkbox("Validación de SPEI", key="SPEI")
            valido_foto = st.selectbox("Oferta corresponde al tamaño del negocio", ["Si", "No", "No se aprecía"],key = "FOTO")

            # Validaciones de campos obligatorios
            campos_validos = (
                Solicitud and
                valido_ine and
                llamada and 
                valido_domicilio and
                valido_curp and
                valido_rfc and
                valido_foto and
                nombre and len(nombre) >= 3 and
                Edad and
                CP and len(CP) == 5 and not resultadosCP.empty and
                genero and
                Oferta and
                Dependientes and
                Edo_civil and
                (Tipo_negocio != "SIN INFORMACION" or (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado)) and
                BimboID and
                blmId and len(blmId) >= 3 and not resultados.empty
            )
            

        submit_button = st.form_submit_button("Guardar registro")

        if submit_button:
            if not campos_validos:
                st.warning("⚠️ Por favor, completa todos los campos correctamente antes de enviar.")
            else:
                try:
                    df = cargar_base()
                    
                    # Obtener Score y Casa si se encontró el BimboID
                    Morosidad_Promedio = resultados.iloc[0]["Morosidad_Promedio"] if not resultados.empty else None
                    Gradient_Boosting_Proba = resultados.iloc[0]["Gradient Boosting_Proba"] if not resultados.empty else None
                    Decil_ventas = resultados.iloc[0]["Decil_ventas"] if not resultados.empty else None
                    PromedioVisitasXMesBimbo = resultados.iloc[0]["PromedioVisitasXMesBimbo"] if not resultados.empty else None
                    ventaPromedioSemanalUlt12Semanas = resultados.iloc[0]["ventaPromedioSemanalUlt12Semanas"] if not resultados.empty else None
                    Giro_de_Cliente = resultados.iloc[0]["Giro_de_Cliente"] if not resultados.empty else None
                    MontoMinCredito = resultados.iloc[0]["MontoMinCredito"] if not resultados.empty else None
                    DiasConCreditoVigente = resultados.iloc[0]["DiasConCreditoVigente"] if not resultados.empty else None
                    Estado = resultadosCP.iloc[0]["d_estado"] if not resultadosCP.empty else None
                    Municipio = resultadosCP.iloc[0]["D_mnpio"] if not resultadosCP.empty else None
                    Ingreso_empleado = resultadosCP.iloc[0]["Ingreso_empleado"] if not resultadosCP.empty else None
                    Morosidad = resultadosCP.iloc[0]["Morosidad"] if not resultadosCP.empty else None

                    nuevo = {
                        'Usuario_registro': str(st.session_state['usuario_actual']),
                        'Solicitud':str(Solicitud),
                        'nombre': nombre,
                        'genero': genero,
                        'Oferta': Oferta,
                        'Información de Crédito': InfoCre,
                        'MicroScore' : MicroScore,
                        'ScoreFico' : ScoreFico,
                        'Dependientes' : Dependientes,
                        'Estado civil' : Edo_civil,
                        'Tipo de negocio' : tipo_negocio_especificado if (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado) else Tipo_negocio,
                        'BimboID': str(BimboID),
                        'blmId' : str(blmId),
                        'Estado de Residencia': Estado,
                        'Municipio de Residencia': Municipio,
                        'Morosidad' : Morosidad,
                        'Morosidad_Promedio': Morosidad_Promedio,
                        'Ingreso_corriente' : Ingreso_empleado,
                        'Edad': Edad,
                        'Gradient Boosting_Proba': Gradient_Boosting_Proba,
                        'Decil_riesgos': asignar_decil(ScoreFico, MicroScore),
                        'PromedioVisitasXMesBimbo': PromedioVisitasXMesBimbo,
                        'Decil_ventas': Decil_ventas,
                        'DiasConCreditoVigente': DiasConCreditoVigente,
                        'ventaPromedioSemanalUlt12Semanas': ventaPromedioSemanalUlt12Semanas,
                        'MontoMinCredito': MontoMinCredito,
                        'Giro_de_Cliente': Giro_de_Cliente,
                        'Validacion_INE' : valido_ine,
                        'Validacion_domicilio' : valido_domicilio,
                        'Validacion_curp' : valido_curp,
                        'Validacion_rfc' : valido_rfc,
                        'Validacion_spei' : valido_spei,
                        'Validacion_foto' : valido_foto,
                        'Comentarios' : Comentarios,
                        'Llamada' : llamada
                    }

                    datos_nuevos = pd.DataFrame([nuevo])

                    # Preprocesar los datos para calificarlos con el modelo de regresion
                    datos_preprocesados = preprocesar_nuevos_datos(datos_nuevos, modelo_cargado)
                    
                    datos_preprocesados['const'] = 1  # Añadir una constante para el intercepto

                    probabilidades = predecir_probabilidades(datos_preprocesados, modelo_cargado)

                    Decil = obtener_decil(probabilidades[0])

                    datos_nuevos["Probabilidad"] = probabilidades
                    datos_nuevos["Decil_modelo"] = Decil

                    ruta_modelo = "AISMaster_Modelo_20241223131859.Rdata"

                    resultadosAIS["EDAD"] = Edad 

                    # Aplica el modelo a tu DataFrame resultadosAIS
                    
                    with st.spinner("Ejecutando modelo AIS en R..."):
                        try:
                            resultado = ejecutar_modelo_ais(
                                nombre_muestra=resultadosAIS,
                                nombre_modelo='AISMaster_Modelo_20241223131859.Rdata'
                            )
                            st.success("✅ Modelo AIS ejecutado correctamente.")
                            st.dataframe(resultado.iloc[:, 2])  # Asegúrate que tiene al menos 3 columnas
                    
                        except Exception as e:
                            st.error(f"❌ Error inesperado durante la ejecución del modelo AIS: {e}")
                            st.stop()
                    print("Se corrio bien modelo AIS")

                    a = float(resultado.iloc[0, 2])

                    Decil_AIS = obtener_decil_AIS(a)
                    print("Se corrio bien decilAIS")
                    datos_nuevos["Probabilidad_AIS"] = a
                    datos_nuevos["Decil_AIS"] = Decil_AIS

                    grupo_nombre, grupo_num = calificar_credito(prob_rl=float(probabilidades.iloc[0]), prob_xgb=a, deciles_rl = deciles_rl, deciles_xgb = deciles_xgb)
                    
                    datos_nuevos["Grupo_nombre"] = grupo_nombre
                    datos_nuevos["Grupo_numero"] = grupo_num

                    Desiscion = asignar_desiscion(grupo_num, MicroScore)

                    datos_nuevos['Desiscion'] = Desiscion

                    print(Desiscion)

                    Min_oferta, Max_oferta = montos_grupo(grupo_num, Desiscion)

                    datos_nuevos['Oferta_min'] = Min_oferta

                    datos_nuevos['Oferta_max'] = Max_oferta

                    Oferta_real = oferta_final(Min_oferta, Max_oferta, Oferta)

                    print("Se asigno la funcion de oferta_final")

                    datos_nuevos['Oferta_final'] = Oferta_real

                    print("Se asigno bien desicion y la oferta real")
                

                    df = pd.concat([df, datos_nuevos], ignore_index=True)
                    guardar_base(df)
                    st.success("✅ Registro guardado correctamente en AWS_S3.")

                    # Guardamos la info en el estado de sesión para mostrar después
                    st.session_state['mostrar_resultado'] = True
                    st.session_state['solicitud_guardada'] = Solicitud
                    st.session_state['nombre_guardado'] = nombre
                    st.session_state['blmId_guardado'] = str(blmId)
                    st.session_state['Desicion_guardada'] = str(Desiscion)
                    st.session_state['Oferta_final'] = Oferta_real
                    
                    # Agregar bandera para indicar que se debe limpiar el formulario
                    st.session_state['limpiar_formulario'] = True
                    
                    # Recargar la página para mostrar los campos limpios
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al guardar el registro: {e}")

    # Mostrar el contenedor con el resultado si existe
    if 'mostrar_resultado' in st.session_state and st.session_state['mostrar_resultado']:
        with st.container():
            st.markdown("### Resultado de la Solicitud")
            st.markdown(f"""
            **Solicitud:** {st.session_state['solicitud_guardada']}  
            **Nombre:**  {st.session_state['nombre_guardado']}  
            **blmId:** {st.session_state['blmId_guardado']}   
            **Oferta:** ${st.session_state['Oferta_final']:,.0f}
            """)
            
            # Mostrar interpretación visual de la probabilidad
            Desicion_value = st.session_state['Desicion_guardada']
            if Desicion_value == "Aceptado":
                st.success(f"🟢 Aceptado")
            else:
                st.error(f"🔴 Rechazado")

    st.markdown("---")
    st.subheader("📊 Registros guardados")

    try:
        base_actual = cargar_base()
        if not base_actual.empty:
            st.dataframe(base_actual)
        else:
            st.info("No hay registros guardados todavía.")
    except Exception as e:
        st.error(f"❌ Error al cargar la base: {e}")

    # Campo de entrada para el ID del cliente a eliminar
    st.markdown("---")
    id_cliente_eliminar = st.text_input("Introduce el ID del cliente a eliminar:")

    if st.button("❌ Eliminar Cliente"):
        if id_cliente_eliminar:
            try:
                df = cargar_base()
                if id_cliente_eliminar in df['Solicitud'].values:
                    eliminar_registro(id_cliente_eliminar)
                    st.success(f"✅ La solicitud {id_cliente_eliminar} ha sido eliminado.")
                    st.rerun()  
                else:
                    st.error(f"No se encontró la solicitud {id_cliente_eliminar}.")
            except Exception as e:
                st.error(f"❌ Error al eliminar la solicitud: {e}")
        else:
            st.warning("Por favor, ingresa una solicitud para eliminar.")
    # Botón de cierre de sesión
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['autenticado'] = False
        st.session_state['usuario_actual'] = ""
        st.rerun()




