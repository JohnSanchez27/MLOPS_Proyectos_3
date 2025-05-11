import os
import sys
import pandas as pd
import mlflow.pyfunc
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException


from datetime import datetime
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, Integer, Float, String, DateTime

# Configuración desde variables de entorno
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_serv:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "mejor_modelo_diabetes")

# Cargar modelo desde MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MLFLOW_MODEL_NAME}/Production"
#model = mlflow.pyfunc.load_model(model_uri)


# Inicialización de FastAPI
app = FastAPI()
# Conexión a base de datos RAW_DATA
from connections import connectionsdb
rawdatadb_engine = connectionsdb[0]

# Esquema de entrada
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

# Endpoint de predicción
@app.post("/predict")
def predict(data: InputData):
    try:

        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        # Verificar si el modelo soporta predict_proba
        try:
            probability = model.predict_proba(input_df)[0][1]
        except AttributeError:
            probability = None

        # Agregar columnas para guardar
        input_df["predicted_readmitted"] = int(prediction[0])
        input_df["prediction_proba"] = float(probability) if probability is not None else None
        input_df["prediction_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Crear tabla si no existe
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


        # Guardar en la base de datos
        input_df.to_sql("predictions", con=rawdatadb_engine, if_exists='append', index=False)

        return {
            "status": 200,
            "prediction": int(prediction[0]),
            "probability": float(probability) if probability is not None else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    