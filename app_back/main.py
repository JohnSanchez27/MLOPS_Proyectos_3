import sys
import os
from pydantic import BaseModel, Field
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
from datetime import datetime
from sqlalchemy import inspect
from math import ceil
from fastapi import FastAPI, HTTPException, status, Query
from typing import Optional
import dags.carga_datos as carga_datos

from dags.preprocesar_datos import preprocesar_datos
from dags.train_model import train_all_batches_and_select_best
from dags.train_model import train_model

from sqlalchemy import inspect,Table, MetaData, Column, Float, Integer, String, DateTime, Text
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

        inspector = inspect(cleandatadb_engine)
        tablas_existentes = inspector.get_table_names()

        # Crear las tablas si no existen, incluyendo batch_id para train_data
        if 'train_data' not in tablas_existentes:
            temp = df_train.copy()
            temp['batch_id'] = 0  # Asegura que se cree la columna
            temp.head(0).to_sql('train_data', con=cleandatadb_engine, index=False, if_exists='replace')
            logging.info("Tabla 'train_data' creada.")
        if 'val_data' not in tablas_existentes:
            df_val.head(0).to_sql('val_data', con=cleandatadb_engine, index=False, if_exists='replace')
            logging.info("Tabla 'val_data' creada.")
        if 'test_data' not in tablas_existentes:
            df_test.head(0).to_sql('test_data', con=cleandatadb_engine, index=False, if_exists='replace')
            logging.info("Tabla 'test_data' creada.")

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

        # Insertar validación y prueba
        df_val.to_sql('val_data', con=cleandatadb_engine, if_exists='append', index=False)
        df_test.to_sql('test_data', con=cleandatadb_engine, if_exists='append', index=False)
        logging.info("Datos de validación y prueba almacenados correctamente.")

        # Consultar totales
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
        train_all_batches_and_select_best()
        return {
            "status": 200,
            "message": "Todos los batches fueron entrenados y el mejor modelo fue copiado como 'modelo_entrenado_final.pkl'."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class InputData(BaseModel):
    race: str = Field(default="Caucasian")
    gender: str = Field(default="Female")
    age: int = Field(default=65)
    time_in_hospital: int = Field(default=3)
    num_lab_procedures: int = Field(default=44)
    num_procedures: int = Field(default=0)
    num_medications: int = Field(default=13)
    number_outpatient: int = Field(default=0)
    number_emergency: int = Field(default=0)
    number_inpatient: int = Field(default=0)
    number_diagnoses: int = Field(default=9)
    max_glu_serum: str = Field(default="None")
    A1Cresult: str = Field(default="None")
    change: str = Field(default="No")
    diabetesMed: str = Field(default="Yes")
    admission_type_id: str = Field(default="Emergencia")
    discharge_disposition_id: str = Field(default="Alta a casa")
    admission_source_id: str = Field(default="Referencia")
    examide: str = Field(default="No")
    citoglipton: str = Field(default="No")

@app.post("/predict")
def predict(data: InputData):
    try:
        # Cargar el mejor modelo entrenado
        try:
            modelo = joblib.load("models/modelo_entrenado_final.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Modelo final no encontrado")

        # Convertir input a DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Realizar predicción
        prediction = modelo.predict(input_df)[0]
        probability = modelo.predict_proba(input_df)[0][1]

        # Agregar columnas para guardar
        input_df["predicted_readmitted"] = int(prediction)
        input_df["prediction_proba"] = round(float(probability), 4)
        input_df["prediction_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Crear tabla 'predictions' si no existe
        inspector = inspect(rawdatadb_engine)
        if "predictions" not in inspector.get_table_names():
            metadata = MetaData()
            columns = []

            for col, dtype in input_df.dtypes.items():
                if dtype == "int64":
                    columns.append(Column(col, Integer))
                elif dtype == "float64":
                    columns.append(Column(col, Float))
                elif dtype.name.startswith("datetime"):
                    columns.append(Column(col, DateTime))
                else:
                    columns.append(Column(col, String(255)))

            Table("predictions", metadata, *columns)
            metadata.create_all(rawdatadb_engine)
            print("Tabla 'predictions' creada.")

        # Guardar predicción
        input_df.to_sql("predictions", con=rawdatadb_engine, if_exists='append', index=False)

        return {
            "status": 200,
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "message": "Predicción realizada y almacenada en raw_data.predictions"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

