import os
import sys
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from datetime import datetime
from fastapi import FastAPI, HTTPException
from connections import connectionsdb
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, DateTime, inspect

# Configuración desde variables de entorno
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_serv:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "mejor_modelo_diabetes")

# Configurar acceso a MinIO (S3-compatible)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "admin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "supersecret")

# Función para cargar el modelo desde el stage Production
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    production_versions = [mv for mv in model_versions if mv.current_stage == 'Production']

    if not production_versions:
        raise ValueError(f"No hay versión en producción para el modelo: {MODEL_NAME}")

    version = production_versions[0].version
    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Modelo '{MODEL_NAME}' versión {version} cargado correctamente.")
    return model

# Intentar cargar el modelo
try:
    model = load_model()
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Inicializar FastAPI
app = FastAPI()

# Conexión a la base de datos RAW_DATA
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
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. No se pudo cargar desde MLflow."
        )
    
    try:

        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        try:
            probability = model.predict_proba(input_df)[0][1]
        except AttributeError:
            probability = None


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

        # Guardar en base de datos
        input_df.to_sql("predictions", con=rawdatadb_engine, if_exists='append', index=False)

        return {
            "status": 200,
            "prediction": int(prediction[0]),
            "probability": float(probability) if probability is not None else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
