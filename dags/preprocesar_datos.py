import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from connections import connectionsdb
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.sensors.time_delta import TimeDeltaSensor

from math import ceil
from sqlalchemy import text, inspect
from sklearn.model_selection import train_test_split


def get_engines():
    from connections import connectionsdb
    rawdatadb_engine = connectionsdb[0]
    cleandatadb_engine = connectionsdb[1]
    return rawdatadb_engine, cleandatadb_engine


def preprocesar_datos():

    rawdatadb_engine, _ = get_engines()
    query = "SELECT * FROM initial_data"
    df = pd.read_sql(query, rawdatadb_engine)


    df = df.replace("?", np.nan)


    df.drop([
        'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'insulin'
    ], axis=1, inplace=True)



    df = df.replace({"NO": 0, "<30": 1, ">30": 0})


    df = df[df["gender"] != "Unknown/Invalid"]


    df['age'] = df['age'].replace({
        "[70-80)": 75, "[60-70)": 65, "[50-60)": 55, "[80-90)": 85,
        "[40-50)": 45, "[30-40)": 35, "[90-100)": 95, "[20-30)": 25,
        "[10-20)": 15, "[0-10)": 5
    })


    df['admission_type_id'] = df['admission_type_id'].replace({
        1.0: "Emergencia", 2.0: "Emergencia", 3.0: "Electiva",
        4.0: "Recién Nacido", 5.0: np.nan, 6.0: np.nan,
        7.0: "Centro de Trauma", 8.0: np.nan
    })

    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace({
        1: "Alta a casa", 6: "Alta a casa", 8: "Alta a casa", 13: "Alta a casa", 19: "Alta a casa",
        18: np.nan, 25: np.nan, 26: np.nan,
        **{k: "Otro" for k in range(2, 31) if k not in [6, 8, 13, 18, 25, 26]}
    })

    df['admission_source_id'] = df['admission_source_id'].replace({
        1: "Referencia", 2: "Referencia", 3: "Referencia", 4: "Otro", 5: "Otro", 6: "Otro",
        10: "Otro", 22: "Otro", 25: "Otro", 9: "Otro", 8: "Otro", 14: "Otro", 13: "Otro",
        11: "Otro", 15: np.nan, 17: np.nan, 20: np.nan, 21: np.nan, 7: "Emergencia"
    })


    for col in ['race', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        df[col] = df[col].fillna(df[col].mode()[0])


    df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)


    df_train, df_rest = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_rest, test_size=0.25, random_state=42)

    return df_train, df_val, df_test

def crear_y_reemplazar_si_existe(df, table_name, engine):
    inspector = inspect(engine)
    tablas = inspector.get_table_names()

    with engine.begin() as conn:
        if table_name not in tablas:
            print(f"Creando tabla {table_name}...")
            df.head(0).to_sql(table_name, con=engine, index=False, if_exists='replace')
        else:
            print(f"Limpiando tabla {table_name}...")
            conn.execute(text(f"DELETE FROM {table_name}"))

    df.to_sql(table_name, con=engine, index=False, if_exists='append')
    print(f"Datos insertados en {table_name}.")

def almacenar_en_clean_data(df_train, df_val, df_test, batch_size=15000, random_state=42):
    _, cleandatadb_engine = get_engines()
    try:
        
        df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Validar existencia y limpiar o crear tabla con estructura + batch_id
        inspector = inspect(cleandatadb_engine)
        tablas = inspector.get_table_names()
        if "train_data" in tablas:
            with cleandatadb_engine.begin() as conn:
                print("Limpiando tabla train_data...")
                conn.execute(text("DELETE FROM train_data"))
        else:
            print("Creando tabla train_data...")
            df_train.head(0).assign(batch_id=0).to_sql("train_data", con=cleandatadb_engine, index=False, if_exists='replace')

        # Insertar por batches con log claro
        total_batches = ceil(len(df_train) / batch_size)

        for batch_number in range(1, total_batches + 1):
            start = (batch_number - 1) * batch_size
            end = batch_number * batch_size
            df_batch = df_train.iloc[start:end].copy()

            if df_batch.empty:
                print(f"Batch {batch_number} está vacío. Se omite.")
                continue

            df_batch["batch_id"] = batch_number
            df_batch.to_sql("train_data", con=cleandatadb_engine, index=False, if_exists='append')
            print(f"Batch {batch_number}/{total_batches} insertado con {len(df_batch)} registros.")

        crear_y_reemplazar_si_existe(df_val, "val_data", cleandatadb_engine)
        crear_y_reemplazar_si_existe(df_test, "test_data", cleandatadb_engine)

        print("Carga en CLEAN_DATA completada.")
        return total_batches

    except Exception as e:
        print(f"Error al almacenar en CLEAN_DATA: {e}")
        raise e

# Definición del DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def ejecutar_preprocesamiento():
    df_train, df_val, df_test = preprocesar_datos()
    almacenar_en_clean_data(df_train, df_val, df_test)
    
with DAG(
    dag_id='preprocesar_datos',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime.now(),
    catchup=False,
    tags=['ETL']
) as dag:


    tarea_preprocesar = PythonOperator(
        task_id='preprocesar_datos',
        python_callable=ejecutar_preprocesamiento
    )

