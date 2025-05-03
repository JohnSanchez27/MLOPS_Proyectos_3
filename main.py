from fastapi import FastAPI,HTTPException,status
from dags.train_model import train_model
import dags.carga_datos as carga_datos

import logging


app = FastAPI()


@app.post("/preprocess")
def preprocess():
    try:
        logging.info("Iniciando el preprocesamiento de datos")
        pipe, X_test, y_test = train_model()
        logging.info(f"Modelo entrenado y validado con éxito: X_test {len(X_test)} y_test {len(y_test)}")
        return {
            "status": status.HTTP_200_OK,
            "message": "Modelo entrenado y validado con éxito"
            }
    except Exception as e:
        logging.error(f"Error en el preprocesamiento de datos: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error en el preprocesamiento de datos")
  
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