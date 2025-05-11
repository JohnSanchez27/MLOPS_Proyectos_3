import sys
import os
import pandas as pd
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from connections import connectionsdb
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.time_delta import TimeDeltaSensor
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from sqlalchemy import create_engine, text, Table, Column, Integer, Float, MetaData, Text, inspect # Crear bases de datos si no existen antes de hacer cualquier cosa

PASSWORD = 'Compaq*87'
def crear_bases_si_no_existen():
    root_engine = create_engine(f'mysql+pymysql://root:{PASSWORD}@localhost:3306')
    with root_engine.connect() as conn:
        conn.execute(text("CREATE DATABASE IF NOT EXISTS RAW_DATA"))
        conn.execute(text("CREATE DATABASE IF NOT EXISTS CLEAN_DATA"))
    print("Bases de datos verificadas/creadas.")

rawdatadb_engine = connectionsdb[0]

def descargar_datos():

    ruta_datos = './data/Diabetes'


    ruta_archivo = os.path.join(ruta_datos, 'Diabetes.csv')


    os.makedirs(ruta_datos, exist_ok=True)


    if not os.path.isfile(ruta_archivo):

        url = 'https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'

        try:

            r = requests.get(url, allow_redirects=True, stream=True)


            if r.status_code == 200:

                with open(ruta_archivo, 'wb') as archivo:
                    archivo.write(r.content)

                if os.path.getsize(ruta_archivo) == 0:
                    raise HTTPException(status_code=500, detail="El archivo descargado está vacío.")
                print(f"Archivo descargado correctamente: {ruta_archivo}")
            else:

                raise HTTPException(status_code=500, detail="Error en la descarga del archivo")
            

        except requests.exceptions.RequestException as e:

            raise HTTPException(status_code=500, detail="Fallo en la conexión al descargar el archivo")
        
    else:

        print(f"El archivo de datos ya existe en: {ruta_archivo}")

    if not os.path.isfile(ruta_archivo) or os.path.getsize(ruta_archivo) == 0:
        raise HTTPException(status_code=500, detail="Archivo CSV no válido o vacío.")

    insert_data(ruta_archivo)

def crear_si_no_existe_y_reemplazar_con_datos(df, table_name, engine):
    inspector = inspect(engine)
    tablas = inspector.get_table_names()
    with engine.begin() as conn:
        if table_name not in tablas:
            print(f"Creando tabla '{table_name}'...")
            df.head(0).to_sql(table_name, con=engine, index=False, if_exists='replace')
        else:
            print(f"Limpiando contenido de '{table_name}'...")
            conn.execute(text(f"DELETE FROM {table_name}"))
    df.to_sql(table_name, con=engine, index=False, if_exists='append')
    print(f"Datos insertados en '{table_name}'.")

def insert_data(ruta_archivo_csv):


    try:
        df = pd.read_csv(ruta_archivo_csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo CSV: {e}")
    
    if df.empty or df.columns.size == 0:
        raise HTTPException(status_code=500, detail="El archivo CSV está vacío o sin columnas.")
    print(f"Columnas del archivo CSV: {list(df.columns)}")
    crear_si_no_existe_y_reemplazar_con_datos(df, "initial_data", rawdatadb_engine)


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='cargar_datos',
    default_args=default_args,
    #schedule_interval='@daily',  
    #start_date=datetime(2025, 5, 1),  
    #catchup=False,  # 
    tags=['Download_upload']
) as dag:

    tarea_descargar = PythonOperator(
        task_id='descargar_datos',
        python_callable=descargar_datos
    )