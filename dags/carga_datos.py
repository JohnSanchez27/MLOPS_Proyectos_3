import os
import requests
from sqlalchemy import create_engine, Table, Column, Integer,String, Float, MetaData,Text
import pandas as pd
from fastapi import HTTPException, status

from connections import connectionsdb

rawdatadb_engine = connectionsdb[0]

def descargar_datos():
    """
    Esta función descarga un conjunto de datos desde una URL si no se encuentra disponible
    localmente en el directorio especificado. Si el archivo ya existe, no lo vuelve a descargar.
    """

    # Definir el directorio raíz para los datos (si no existe, se creará)
    ruta_datos = './data/Diabetes'

    # Ruta completa al archivo de datos sin procesar
    ruta_archivo = os.path.join(ruta_datos, 'Diabetes.csv')

    # Crear el directorio para los datos si no existe
    os.makedirs(ruta_datos, exist_ok=True)

    # Verificar si el archivo de datos ya existe en la ruta especificada
    if not os.path.isfile(ruta_archivo):
        # URL de descarga directa del conjunto de datos
        # Nota: Es importante actualizar la URL si se cambia el enlace
        url = 'https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
        
        # Hacer la solicitud HTTP para descargar el archivo
        try:
            r = requests.get(url, allow_redirects=True, stream=True)

            # Verificar que la respuesta de la solicitud sea exitosa (código 200)
            if r.status_code == 200:
                # Escribir el contenido descargado en un archivo local
                with open(ruta_archivo, 'wb') as archivo:
                    archivo.write(r.content)
                print(f"El archivo de datos se ha descargado correctamente en: {ruta_archivo}")
            else:
                print(f"Error al descargar el archivo. Código de estado: {r.status_code}")

        except requests.exceptions.RequestException as e:
            # Manejo de excepciones si hay problemas con la descarga
            print(f"Hubo un problema al intentar descargar el archivo: {e}")
    else:
        # Si el archivo ya existe, se omite la descarga
        print(f"El archivo de datos ya existe en: {ruta_archivo}")

    insert_data(ruta_archivo)

def insert_data(ruta_archivo_csv):
    metada_raw = MetaData()

    initial_data_table = Table('initial_data', metada_raw,)
    
    df = pd.read_csv(ruta_archivo_csv)
    headers = list(df.columns)
    
    print(f"Columnas del archivo CSV: {headers}")
    print("Agregar columna 0 como primary key")
    initial_data_table.append_column(Column(headers[0], Integer, primary_key=True))

    print("Agregando columnas al esquema de la tabla") 
    id_flag = False
    for column_name in headers:
        if id_flag == False:
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
            # Insertar los datos en la tabla
            df.to_sql('initial_data', con=connection, if_exists='replace', index=False)
            print("Datos insertados correctamente en la tabla 'initial_data'")
    except Exception as e:
        print(f"Error al insertar datos en la tabla: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error en la carga de datos")
    
        
    