# Proyecto 3 - MLOps (Fase Local)

Este proyecto implementa una arquitectura básica de MLOps de manera local para el procesamiento de datos, entrenamiento y predicción de un modelo de clasificación, utilizando FastAPI y almacenamiento en MySQL.

---

## 📁 Estructura del Proyecto

Proyecto_3/
│
├── app_back/ # Backend con FastAPI
│ └── main.py # Endpoints de carga, procesamiento, entrenamiento y predicción
│
├── connections/
│ ├── init.py
│ └── mysql_connections.py # Configuración de conexión a MySQL (RAW y CLEAN)
│
├── dags/
│ ├── carga_datos.py # Inserta datos crudos en RAW
│ ├── preprocesar_datos.py # Procesa datos y los guarda por lotes en CLEAN
│ └── train_model.py # Entrena modelo, guarda .pkl y métricas
│
├── data/
│ └── Diabetes.csv # Dataset original
│
├── modelo_entrenado.pkl # Modelo serializado entrenado (Random Forest)
├── requirements.txt # Dependencias del proyecto
└── README.md # Documentación del proyecto


---

## ✅ Funcionalidad Actual

- **Carga de datos**  
  Inserta datos crudos en la tabla `initial_data` del esquema `raw_data`.

- **Preprocesamiento**  
  Limpia los datos y los divide en train, validation y test. El conjunto `train` se carga por lotes en `clean_data.train_data`.

- **Entrenamiento del modelo**  
  Entrena un modelo `RandomForestClassifier` usando `Pipeline` de `scikit-learn`, almacena el modelo como `modelo_entrenado.pkl`, y registra métricas (`accuracy`, `precision`, `recall`, `f1-score`) en la tabla `clean_data.experiments`.

- **Predicción**  
  Utiliza el modelo entrenado para predecir nuevas muestras y guarda las predicciones en `raw_data.predictions`.

- **API REST con FastAPI**  
  Permite ejecutar cada etapa del flujo mediante endpoints:

| Endpoint        | Acción                                      |
|----------------|---------------------------------------------|
| `/download`     | Descarga el dataset e inserta en RAW       |
| `/clean`        | Preprocesa y divide los datos               |
| `/train`        | Entrena el modelo y guarda métricas         |
| `/predict`      | Realiza predicciones con el modelo entrenado |

---

## 🧰 Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt

