import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocesar_datos():
    
    # Leer el archivo local CSV descargado
    ruta_archivo = './notebooks/data/Diabetes/Diabetes.csv'
    df = pd.read_csv(ruta_archivo)

    # Reemplazar valores desconocidos con NaN
    df = df.replace("?", np.nan)

    # Eliminar columnas no necesarias
    df.drop(['weight','payer_code','medical_specialty','diag_1', 'diag_2', 'diag_3', 'metformin','repaglinide', 
             'nateglinide', 'chlorpropamide', 'glimepiride','acetohexamide', 'glipizide','glyburide', 'tolbutamide',
             'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone','tolazamide', 
             'glyburide-metformin', 'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone',
             'metformin-pioglitazone', 'insulin'], axis=1, inplace=True)

    # Codificar la variable objetivo
    df = df.replace({"NO": 0, "<30": 1, ">30": 0})

    # Eliminar registros con género desconocido
    df = df.drop(df.loc[df["gender"] == "Unknown/Invalid"].index, axis=0)

    # Reemplazar intervalos de edad por promedios
    df.age = df.age.replace({"[70-80)": 75, "[60-70)": 65, "[50-60)": 55,
                             "[80-90)": 85, "[40-50)": 45, "[30-40)": 35,
                             "[90-100)": 95, "[20-30)": 25, "[10-20)": 15, "[0-10)": 5})

    # Mapear valores de tipo de admisión
    mapeo_admision = {1.0: "Emergencia", 2.0: "Emergencia", 3.0: "Electiva",
                      4.0: "Recién Nacido", 5.0: np.nan, 6.0: np.nan,
                      7.0: "Centro de Trauma", 8.0: np.nan}
    df.admission_type_id = df.admission_type_id.replace(mapeo_admision)

    # Mapear valores de disposición al alta
    mapeo_alta = {1: "Alta a casa", 6: "Alta a casa", 8: "Alta a casa", 13: "Alta a casa", 19: "Alta a casa",
                  18: np.nan, 25: np.nan, 26: np.nan,
                  2: "Otro", 3: "Otro", 4: "Otro", 5: "Otro", 7: "Otro", 9: "Otro",
                  10: "Otro", 11: "Otro", 12: "Otro", 14: "Otro", 15: "Otro", 16: "Otro",
                  17: "Otro", 20: "Otro", 21: "Otro", 22: "Otro", 23: "Otro", 24: "Otro",
                  27: "Otro", 28: "Otro", 29: "Otro", 30: "Otro"}
    df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(mapeo_alta)

    # Mapear fuente de admisión
    mapeo_fuente = {1: "Referencia", 2: "Referencia", 3: "Referencia",
                    4: "Otro", 5: "Otro", 6: "Otro", 10: "Otro", 22: "Otro", 25: "Otro",
                    9: "Otro", 8: "Otro", 14: "Otro", 13: "Otro", 11: "Otro",
                    15: np.nan, 17: np.nan, 20: np.nan, 21: np.nan,
                    7: "Emergencia"}
    df.admission_source_id = df.admission_source_id.replace(mapeo_fuente)

    # Completar valores faltantes con la moda
    df['race'] = df['race'].fillna(df['race'].mode()[0])
    df['admission_type_id'] = df['admission_type_id'].fillna(df['admission_type_id'].mode()[0])
    df['discharge_disposition_id'] = df['discharge_disposition_id'].fillna(df['discharge_disposition_id'].mode()[0])
    df['admission_source_id'] = df['admission_source_id'].fillna(df['admission_source_id'].mode()[0])

    # Eliminar columnas de identificadores
    df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

    # Información general
    print(df.info())

    # División de los datos en entrenamiento, validación y prueba
    df_train, df_rest = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_rest, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    print("Tamaño df_train:", df_train.shape)
    print("Tamaño df_val:", df_val.shape)
    print("Tamaño df_test:", df_test.shape)

    return df_train, df_val, df_test

preprocessdata = preprocesar_datos