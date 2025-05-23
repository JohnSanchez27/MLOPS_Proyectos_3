import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

from sqlalchemy import text, inspect, Table, MetaData, Column, Float, Integer, DateTime
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Encapsular la conexión
def get_engine():
    from connections import connectionsdb
    return connectionsdb[1]

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
    engine = get_engine()
    query = f"SELECT * FROM train_data WHERE batch_id = {batch_id}"
    df_train = pd.read_sql(query, engine)
    df_val = pd.read_sql("SELECT * FROM val_data", engine)
    df_test = pd.read_sql("SELECT * FROM test_data", engine)

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


    mlflow.set_tracking_uri("http://mlflow_serv:5000")
    mlflow.set_experiment("experiment_diabetes_batches")

    with mlflow.start_run(run_name=f"batch_{batch_id}"):
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

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(pipe, artifact_path="modelo")

        crear_tabla_experiments_si_no_existe(engine)
        insert_query = text("""
            INSERT INTO experiments (
                timestamp, val_accuracy, val_precision, val_recall, val_f1,
                test_accuracy, test_precision, test_recall, test_f1, batch_id
            ) VALUES (
                NOW(), :val_accuracy, :val_precision, :val_recall, :val_f1,
                :test_accuracy, :test_precision, :test_recall, :test_f1, :batch_id
            )
        """)
        with engine.begin() as conn:
            conn.execute(insert_query, metrics)

        model_dir = ensure_model_dir()
        model_path = os.path.join(model_dir, f"modelo_entrenado_batch{batch_id}.pkl")
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(model_path)
        print(f"Modelo guardado como {model_path}")

    return metrics, model_path

def train_all_batches_and_select_best(metric="val_f1"):
    engine = get_engine()
    query = "SELECT DISTINCT batch_id FROM train_data ORDER BY batch_id"
    batch_ids = pd.read_sql(query, engine)["batch_id"].tolist()

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

        # Registrar modelo final en MLflow
        mlflow.set_tracking_uri("http://mlflow_serv:5000")
        mlflow.set_experiment("experiment_diabetes_batches")

        with mlflow.start_run(run_name="modelo_final") as run:
            run_id = run.info.run_id  # Se guarda antes de que el contexto se cierre
            mlflow.log_param("best_batch", best_batch)
            mlflow.log_metric(metric, best_score)
            mlflow.log_artifact(final_model_path)
            modelo_final = joblib.load(final_model_path)
            mlflow.sklearn.log_model(modelo_final, artifact_path="modelo_final")
            print("Modelo final registrado en MLflow.")

        # Esta parte va *fuera* del with
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/modelo_final",
            name="mejor_modelo_diabetes"
        )


        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="mejor_modelo_diabetes",
            version=result.version,
            stage="Production",
            archive_existing_versions=True
)
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
    start_date=datetime.now(),
    catchup=False,
    tags=['ETL', 'modelo']
) as dag:
    


    entrenar_modelos = PythonOperator(
        task_id='train_model_all_batches',
        python_callable=train_all_batches_and_select_best
    )

    