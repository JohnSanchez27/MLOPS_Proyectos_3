"""Pipeline Diabetes

Este script prepara y entrena un modelo de clasificación usando Random Forest
para predecir la readmisión de pacientes diabéticos.

"""

import pandas as pd
import os
import requests
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# Crear carpeta para los datos y descargar el archivo si no existe
ruta_datos = './data/Diabetes'
ruta_archivo = os.path.join(ruta_datos, 'Diabetes.csv')
os.makedirs(ruta_datos, exist_ok=True)

if not os.path.isfile(ruta_archivo):
    url = 'https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
    r = requests.get(url, allow_redirects=True, stream=True,verify=False)
    open(ruta_archivo, 'wb').write(r.content)

# Cargar los datos
df = pd.read_csv(ruta_archivo, encoding="utf8")

# Reemplazar valores faltantes indicados como "?" con NaN
df = df.replace("?", np.nan)

# Eliminar columnas irrelevantes o con muchos valores faltantes
df.drop(['weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'metformin', 'repaglinide',
         'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
         'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
         'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
         'metformin-pioglitazone', 'insulin'], axis=1, inplace=True)

# Codificar la variable objetivo (readmitted) y otras categorías
df = df.replace({"NO": 0, "<30": 1, ">30": 0})

# Eliminar filas con género no válido
df = df.drop(df[df["gender"] == "Unknown/Invalid"].index, axis=0)

# Convertir los rangos de edad a valores medios
df.age = df.age.replace({
    "[70-80)": 75, "[60-70)": 65, "[50-60)": 55,
    "[80-90)": 85, "[40-50)": 45, "[30-40)": 35,
    "[90-100)": 95, "[20-30)": 25, "[10-20)": 15,
    "[0-10)": 5
})

# Mapear el tipo de admisión a categorías legibles
df.admission_type_id = df.admission_type_id.replace({
    1.0: "Emergencia", 2.0: "Emergencia", 3.0: "Electiva", 4.0: "Recién nacido",
    5.0: np.nan, 6.0: np.nan, 7.0: "Centro de trauma", 8.0: np.nan
})

# Mapear tipo de egreso a categorías legibles
df.discharge_disposition_id = df.discharge_disposition_id.replace({
    1: "Alta a casa", 6: "Alta a casa", 8: "Alta a casa", 13: "Alta a casa", 19: "Alta a casa",
    18: np.nan, 25: np.nan, 26: np.nan,
    **{k: "Otro" for k in [2,3,4,5,7,9,10,11,12,14,15,16,17,20,21,22,23,24,27,28,29,30]}
})

# Mapear la fuente de admisión
df.admission_source_id = df.admission_source_id.replace({
    1: "Referencia", 2: "Referencia", 3: "Referencia",
    4: "Otro", 5: "Otro", 6: "Otro", 10: "Otro", 22: "Otro", 25: "Otro",
    9: "Otro", 8: "Otro", 14: "Otro", 13: "Otro", 11: "Otro",
    15: np.nan, 17: np.nan, 20: np.nan, 21: np.nan,
    7: "Emergencia"
})

# Imputar valores faltantes con la moda
df['race'] = df['race'].fillna(df['race'].mode()[0])
df['admission_type_id'] = df['admission_type_id'].fillna(df['admission_type_id'].mode()[0])
df['discharge_disposition_id'] = df['discharge_disposition_id'].fillna(df['discharge_disposition_id'].mode()[0])
df['admission_source_id'] = df['admission_source_id'].fillna(df['admission_source_id'].mode()[0])

# Eliminar identificadores únicos
df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

# Mostrar conteo de readmitidos y no readmitidos
print(df[df['readmitted'] == 1].shape)
print(df[df['readmitted'] == 0].shape)

# Separar las variables predictoras (X) y la variable objetivo (y)
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Identificar columnas categóricas y numéricas
variables_categoricas = df.select_dtypes('O')
variables_numericas = df.select_dtypes(np.number)

# Clase personalizada para codificar variables categóricas en variables dummy
class ObtenerDummies(BaseEstimator, TransformerMixin):
    def __init__(self, variables_categoricas=None, drop_first=False):
        self.variables_categoricas = variables_categoricas
        self.drop_first = drop_first

    def reemplazar_nombres_invalidos(self, dummies):
        for s in dummies.columns:
            nuevo = re.sub(r'\[', '(', s)
            dummies.rename(columns={s: nuevo}, inplace=True)
        for s in dummies.columns:
            nuevo = re.sub(r'\<', 'menos_que', s)
            dummies.rename(columns={s: nuevo}, inplace=True)
        for s in dummies.columns:
            nuevo = re.sub(r'\>', 'mayor_que', s)
            dummies.rename(columns={s: nuevo}, inplace=True)
        return dummies

    def fit(self, X, y=None):
        self.variables_categoricas = [col for col in X.columns if X[col].dtype.name in ('category', 'object')]
        return self

    def transform(self, X):
        dummies = pd.get_dummies(X, columns=self.variables_categoricas, drop_first=self.drop_first)
        return self.reemplazar_nombres_invalidos(dummies)

# Clase personalizada para escalar los datos numéricos
class EscaladorPersonalizado(BaseEstimator, TransformerMixin):
    def __init__(self, escalador):
        self.escalador = escalador

    def fit(self, X, y=None):
        self.escalador.fit(X)
        return self

    def transform(self, X):
        nuevo_X = self.escalador.transform(X)
        return pd.DataFrame(nuevo_X, columns=X.columns)

# Crear modelo con Random Forest
modelo = RandomForestClassifier()

# Crear el pipeline con codificación, escalado y modelo
preprocesador = Pipeline([
    ('encoder', ObtenerDummies()),
    ('scaler', EscaladorPersonalizado(StandardScaler())),
    ('classifier', modelo)
])

# Dividir los datos en entrenamiento, validación y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
X_entrenamiento, X_val, y_entrenamiento, y_val = train_test_split(X_entrenamiento, y_entrenamiento, test_size=0.2, random_state=42)

# Reiniciar los índices de los conjuntos
X_entrenamiento.reset_index(drop=True, inplace=True)
y_entrenamiento.reset_index(drop=True, inplace=True)
X_prueba.reset_index(drop=True, inplace=True)
y_prueba.reset_index(drop=True, inplace=True)

print(X_entrenamiento.shape)

# Entrenar el pipeline
preprocesador.fit(X_entrenamiento, y_entrenamiento)

# Realizar predicciones sobre el conjunto de prueba
y_predicho = preprocesador.predict(X_prueba)

# Mostrar matriz de confusión y métricas de evaluación
print(confusion_matrix(y_prueba, y_predicho))
print("Precisión:", accuracy_score(y_prueba, y_predicho))
print("Sensibilidad:", recall_score(y_prueba, y_predicho))