import os
import requests
#from airflow import DAG
#from airflow.operators.python_operator import PythonOperator
#from airflow.sensors.time_delta import TimeDeltaSensor

import pandas as pd

from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, Text
from fastapi import HTTPException, status

from connections import connectionsdb

rawdatadb_engine = connectionsdb[0]

def descargar_datos():
    """
    Descarga un conjunto de datos desde una URL si no se encuentra disponible
    localmente. Inserta los datos en la base de datos si son válidos.
    """

    
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
                print(f"Error al descargar el archivo. Código de estado: {r.status_code}")
                raise HTTPException(status_code=500, detail="Error en la descarga del archivo")
            
    
        except requests.exceptions.RequestException as e:
    
            print(f"Hubo un problema al intentar descargar el archivo: {e}")
            raise HTTPException(status_code=500, detail="Fallo en la conexión al descargar el archivo")
    
    else:

        print(f"El archivo de datos ya existe en: {ruta_archivo}")

    if not os.path.isfile(ruta_archivo) or os.path.getsize(ruta_archivo) == 0:
        raise HTTPException(status_code=500, detail="Archivo CSV no válido o vacío.")

    insert_data(ruta_archivo)

def insert_data(ruta_archivo_csv):
    metada_raw = MetaData()
    initial_data_table = Table('initial_data', metada_raw)

    try:
        df = pd.read_csv(ruta_archivo_csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo CSV: {e}")

    if df.empty or df.columns.size == 0:
        raise HTTPException(status_code=500, detail="El archivo CSV está vacío o sin columnas.")

    headers = list(df.columns)

    print(f"Columnas del archivo CSV: {headers}")
    print("Agregar columna 0 como primary key")
    initial_data_table.append_column(Column(headers[0], Integer, primary_key=True))

    print("Agregando columnas al esquema de la tabla")
    id_flag = False
    for column_name in headers:
        if not id_flag:
            id_flag = True
            continue

        print(f"Agregando columna: {column_name} {df[column_name].dtype}")
        if df[column_name].dtype == 'object':
            initial_data_table.append_column(Column(column_name, Text))
        elif df[column_name].dtype == 'int64':
            initial_data_table.append_column(Column(column_name, Integer))
        elif df[column_name].dtype == 'float64':
            initial_data_table.append_column(Column(column_name, Float))
        else:
            initial_data_table.append_column(Column(column_name, Text))

    metada_raw.create_all(rawdatadb_engine, checkfirst=True)

    try:
        with rawdatadb_engine.begin() as connection:
            
            df.to_sql('initial_data', con=connection, if_exists='replace', index=False)
            print("Datos insertados correctamente en la tabla 'initial_data'")
    except Exception as e:
        print(f"Error al insertar datos en la tabla: {e}")
        raise HTTPException(status_code=500, detail="Error en la carga de datos")
