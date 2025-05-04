import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sqlalchemy import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from connections import connectionsdb

# Conexión a la base CLEAN_DATA
cleandatadb_engine = connectionsdb[1]

def train_model():
    # Leer los datos desde la base de datos
    df_train = pd.read_sql("SELECT * FROM train_data", cleandatadb_engine)
    df_val = pd.read_sql("SELECT * FROM val_data", cleandatadb_engine)
    df_test = pd.read_sql("SELECT * FROM test_data", cleandatadb_engine)

    # Separar características y etiquetas
    X_train = df_train.drop(["readmitted", "batch_id"], axis=1)
    y_train = df_train["readmitted"]

    X_val = df_val.drop("readmitted", axis=1)
    y_val = df_val["readmitted"]

    X_test = df_test.drop("readmitted", axis=1)
    y_test = df_test["readmitted"]

    # Identificar columnas categóricas y numéricas
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Definir el preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

    # Crear el pipeline
    model = RandomForestClassifier(random_state=42)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Entrenar el modelo
    pipe.fit(X_train, y_train)

    # Predecir y evaluar
    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    metrics = {
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_precision": precision_score(y_val, val_pred),
        "val_recall": recall_score(y_val, val_pred),
        "val_f1": f1_score(y_val, val_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred),
        "test_recall": recall_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred),
    }

    print("Métricas de validación y test:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Guardar las métricas en la tabla `experiments`
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    insert_query = text("""
        INSERT INTO experiments (
            timestamp, val_accuracy, val_precision, val_recall, val_f1,
            test_accuracy, test_precision, test_recall, test_f1
        ) VALUES (
            :timestamp, :val_accuracy, :val_precision, :val_recall, :val_f1,
            :test_accuracy, :test_precision, :test_recall, :test_f1
        )
    """)
    with cleandatadb_engine.begin() as conn:
        conn.execute(insert_query, {"timestamp": timestamp, **metrics})

    # Guardar modelo como archivo .pkl
    joblib.dump(pipe, "modelo_entrenado.pkl")
    print("Modelo guardado como 'modelo_entrenado.pkl'.")

    return pipe
