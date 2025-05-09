import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from connections import connectionsdb
import shutil
import pandas as pd
import numpy as np
import joblib

import mlflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

from sqlalchemy import text, inspect, Table, MetaData, Column, Float, Integer, DateTime
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




cleandatadb_engine = connectionsdb[1]


def ensure_model_dir():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def crear_tabla_experiments_si_no_existe(engine):
    inspector = inspect(engine)
    if 'experiments' not in inspector.get_table_names():
        metadata = MetaData()
        Table('experiments', metadata,
              Column('timestamp', DateTime),
              Column('val_accuracy', Float),
              Column('val_precision', Float),
              Column('val_recall', Float),
              Column('val_f1', Float),
              Column('test_accuracy', Float),
              Column('test_precision', Float),
              Column('test_recall', Float),
              Column('test_f1', Float),
              Column('batch_id', Integer)
        )
        metadata.create_all(engine)
        print("Tabla 'experiments' creada.")

def train_model(batch_id: int):
    query = f"SELECT * FROM train_data WHERE batch_id = {batch_id}"
    df_train = pd.read_sql(query, cleandatadb_engine)
    df_val = pd.read_sql("SELECT * FROM val_data", cleandatadb_engine)
    df_test = pd.read_sql("SELECT * FROM test_data", cleandatadb_engine)

    
    X_train = df_train.drop(["readmitted", "batch_id"], axis=1)
    y_train = df_train["readmitted"]

    X_val = df_val.drop("readmitted", axis=1)
    y_val = df_val["readmitted"]

    X_test = df_test.drop("readmitted", axis=1)
    y_test = df_test["readmitted"]

    categorical = X_train.select_dtypes(include="object").columns.tolist()
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical)
    ])

    
    model = RandomForestClassifier(random_state=42)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    
    pipe.fit(X_train, y_train)

    
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
        "batch_id": batch_id
    }

    crear_tabla_experiments_si_no_existe(cleandatadb_engine)

    insert_query = text("""
        INSERT INTO experiments (
            timestamp, val_accuracy, val_precision, val_recall, val_f1,
            test_accuracy, test_precision, test_recall, test_f1, batch_id
        ) VALUES (
            NOW(), :val_accuracy, :val_precision, :val_recall, :val_f1,
            :test_accuracy, :test_precision, :test_recall, :test_f1, :batch_id
        )
    """)
    with cleandatadb_engine.begin() as conn:
        conn.execute(insert_query, metrics)

    
    model_dir = ensure_model_dir()
    model_path = os.path.join(model_dir, f"modelo_entrenado_batch{batch_id}.pkl")
    joblib.dump(pipe, model_path)
    print(f"Modelo guardado como {model_path}")

    return metrics, model_path

def train_all_batches_and_select_best(metric="val_f1"):
    query = "SELECT DISTINCT batch_id FROM train_data ORDER BY batch_id"
    batch_ids = pd.read_sql(query, cleandatadb_engine)["batch_id"].tolist()

    resultados = []
    for batch_id in batch_ids:
        try:
            print(f"Entrenando batch {batch_id}...")
            metrics, model_file = train_model(batch_id)
            print(f"Batch {batch_id} entrenado. {metric}={metrics[metric]:.4f}")
            resultados.append((batch_id, metrics[metric], model_file))
        except Exception as e:
            print(f"Error en batch {batch_id}: {e}")

    if not resultados:
        print("No se entrenó ningún batch.")
        return

    resultados.sort(key=lambda x: x[1], reverse=True)
    best_batch, best_score, best_file = resultados[0]

    model_dir = ensure_model_dir()
    final_model_path = os.path.join(model_dir, "modelo_entrenado_final.pkl")

    if os.path.exists(best_file):
        shutil.copy(best_file, final_model_path)
        print(f"Mejor modelo: batch {best_batch} con {metric}={best_score:.4f}")
        print(f"Copiado como {final_model_path}")
    else:
        print(f"No se encontró el archivo {best_file} para copiarlo como modelo final.")

# Definición del DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='entrenar_modelo',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 5, 1),
    catchup=False,
    tags=['ETL', 'modelo']
) as dag:

    esperar_preprocesamiento = ExternalTaskSensor(
        task_id='esperar_preprocesamiento',
        external_dag_id='preprocesar_datos',          # nombre del DAG anterior
        external_task_id='preprocesar_datos',         # task_id que debe terminar
        mode='poke',
        timeout=600,
        poke_interval=30
    )

    entrenar_modelos = PythonOperator(
        task_id='train_model_all_batches',
        python_callable=train_all_batches_and_select_best
    )

    esperar_preprocesamiento >> entrenar_modelos


