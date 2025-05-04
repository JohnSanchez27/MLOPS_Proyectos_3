import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
from datetime import datetime
from math import ceil
from fastapi import FastAPI, HTTPException, status, Query
from typing import Optional
import dags.carga_datos as carga_datos

from dags.preprocesar_datos import preprocesar_datos

from dags.train_model import train_model

from sqlalchemy import text
from connections import connectionsdb

import logging

app = FastAPI()
rawdatadb_engine = connectionsdb[0]  # conexión a RAW_DATA
cleandatadb_engine = connectionsdb[1] # conexión a CLEAN_DATA

@app.get("/download")
def download():
    try:
        logging.info("Iniciando la descarga de datos")
        carga_datos.descargar_datos()
        logging.info("Datos descargados y almacenados correctamente")
        return {
            "status": status.HTTP_200_OK,
            "message": "Datos descargados y almacenados correctamente"
            }
    except Exception as e:
        logging.error(f"Error en la descarga de datos: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error en la descarga de datos")

@app.get("/clean")
def clean():
    try:
        logging.info("Iniciando preprocesamiento y carga de datos en CLEAN_DATA")
        df_train, df_val, df_test = preprocesar_datos()
        batch_size = 15000
        total_batches = ceil(len(df_train) / batch_size)

        # Limpiar las tablas antes de insertar nuevos datos
        with cleandatadb_engine.begin() as conn:
            conn.execute(text("DELETE FROM train_data"))
            conn.execute(text("DELETE FROM val_data"))
            conn.execute(text("DELETE FROM test_data"))

        # Insertar los batches de train
        for batch_number in range(1, total_batches + 1):
            start = (batch_number - 1) * batch_size
            end = batch_number * batch_size
            df_train_batch = df_train.iloc[start:end].copy()

            if df_train_batch.empty:
                logging.warning(f"Batch {batch_number} está vacío. Se omite.")
                continue

            df_train_batch['batch_id'] = batch_number
            df_train_batch.to_sql('train_data', con=cleandatadb_engine, if_exists='append', index=False)
            logging.info(f"Batch {batch_number} almacenado correctamente.")

        # Insertar validación y prueba una sola vez
        df_val.to_sql('val_data', con=cleandatadb_engine, if_exists='replace', index=False)
        df_test.to_sql('test_data', con=cleandatadb_engine, if_exists='replace', index=False)
        logging.info("Datos de validación y prueba almacenados correctamente.")

        with cleandatadb_engine.connect() as conn:
            train_rows = conn.execute(text("SELECT COUNT(*) FROM train_data")).scalar()
            val_rows = conn.execute(text("SELECT COUNT(*) FROM val_data")).scalar()
            test_rows = conn.execute(text("SELECT COUNT(*) FROM test_data")).scalar()

        return {
            "status": 200,
            "message": f"{total_batches} batches cargados correctamente.",
            "train_total_rows": train_rows,
            "val_total_rows": val_rows,
            "test_total_rows": test_rows,
            "batch_size": batch_size
        }

    except Exception as e:
        logging.error(f"Error en el endpoint /clean: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train")
def train_endpoint():
    try:
        modelo = train_model()  # Esto ya entrena, guarda .pkl y registra en MySQL
        return {
            "status": 200,
            "message": "Modelo entrenado, guardado y métricas registradas correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
def predict():
    try:
        # Cargar modelo entrenado
        try:
            modelo = joblib.load("modelo_entrenado.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Modelo entrenado no encontrado")

        # Leer datos desde test_data
        df_test = pd.read_sql("SELECT * FROM test_data", cleandatadb_engine)

        # Separar X
        X_test = df_test.drop("readmitted", axis=1)

        # Predecir
        predictions = modelo.predict(X_test)

        # Agregar predicciones y timestamp al DataFrame original
        df_test["predicted_readmitted"] = predictions
        df_test["prediction_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Guardar en tabla raw_data.predictions
        df_test.to_sql("predictions", con=rawdatadb_engine, if_exists='append', index=False)

        return {
            "status": 200,
            "message": f"{len(predictions)} predicciones realizadas y almacenadas en raw_data.predictions"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    