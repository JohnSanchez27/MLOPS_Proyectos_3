# Proyecto 3 - MLOps (Fase Local)

Este proyecto implementa una arquitectura básica de MLOps de manera local para el procesamiento de datos, entrenamiento y predicción de un modelo de clasificación, utilizando FastAPI y almacenamiento en MySQL.

---

## 📁 Estructura del Proyecto

```text
Proyecto_3/
│
├── app_back/                      # Backend con FastAPI
│   └── main.py                    # Endpoints de carga, procesamiento, entrenamiento y predicción
│
├── app_front/                     # Interfaz Streamlit (opcional)
│   └── app.py
│
├── connections/                   # Configuración de conexión a MySQL (RAW y CLEAN)
│   ├── __init__.py
│   └── mysql_connections.py
│
├── dags/                          # Scripts de procesamiento, carga y entrenamiento
│   ├── carga_datos.py             # Inserta datos crudos en RAW
│   ├── preprocesar_datos.py       # Procesa datos y los guarda por lotes en CLEAN
│   └── train_model.py             # Entrena modelos por lote, guarda métricas y el mejor modelo
│
├── data/Diabetes/                 # Dataset original
│   └── Diabetes.csv
│
├── models/                        # Modelos serializados entrenados
│   ├── modelo_entrenado_batch1.pkl
│   ├── modelo_entrenado_batch2.pkl
│   ├── modelo_entrenado_batch3.pkl
│   ├── modelo_entrenado_batch4.pkl
│   ├── modelo_entrenado_batch5.pkl
│   ├── modelo_entrenado_batch6.pkl
│   └── modelo_entrenado_final.pkl  # Mejor modelo basado
│
├── imagenes/                      # Recursos gráficos
│   ├── arquitectura.png
│   └── Front_Streamlit.png
│
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Documentación del proyecto
```

## ✅ Funcionalidad Actual


- **Carga de datos**  
  Inserta datos crudos en la tabla `initial_data` del esquema `raw_data`.

- **Preprocesamiento**  
  Limpia los datos, elimina columnas irrelevantes, transforma variables categóricas y divide el dataset en conjuntos de `entrenamiento`, `validación` y `prueba`. El conjunto `train` se mezcla aleatoriamente y se guarda por lotes en `clean_data.train_data`.

- **Entrenamiento del modelo**  
  Entrena múltiples modelos `RandomForestClassifier` (uno por cada batch) usando `Pipeline` de `scikit-learn`, guarda cada uno como `modelo_entrenado_batch{n}.pkl`, registra sus métricas (`accuracy`, `precision`, `recall`, `f1-score`) en la tabla `clean_data.experiments` y selecciona el mejor modelo basado en `val_f1`, copiándolo como `modelo_entrenado_final.pkl`.

- **Predicción vía API**  
  Utiliza el modelo `modelo_entrenado_final.pkl` para predecir nuevas muestras y guarda las predicciones con sus probabilidades en `raw_data.predictions`.

- **Interfaz gráfica con Streamlit**  
  Permite a los usuarios realizar predicciones desde un formulario web amigable, que se comunica con la API FastAPI para enviar datos y mostrar resultados.

  ![Interfaz Streamlit](imagenes/Front_Streamlit.png)

- **API REST con FastAPI**  
  Permite ejecutar cada etapa del flujo mediante endpoints:

| Endpoint        | Acción                                                   |
|----------------|----------------------------------------------------------|
| `/download`     | Descarga el dataset e inserta en `raw_data.initial_data` |
| `/clean`        | Preprocesa y divide los datos, luego guarda por lotes    |
| `/train`        | Entrena todos los modelos por lote y selecciona el mejor |
| `/predict`      | Realiza predicciones usando el mejor modelo entrenado    |

---

## 🧰 Requisitos y Puesta en Marcha

Sigue estos pasos para ejecutar el proyecto localmente:

```bash
# 1. Clonar el repositorio
git clone git@github.com:JohnSanchez27/MLOPS_Proyectos_3.git
cd MLOPS_Proyectos_3

# 2. Crear y activar un entorno virtual
# En Windows:
python -m venv .venv
.venv\Scripts\activate
# En Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Asegurar que el servidor MySQL esté activo y con los esquemas raw_data y clean_data creados
#    Editar el archivo de conexión: connections/mysql_connections.py

# 5. Iniciar el backend (FastAPI)
uvicorn app_back.main:app --reload
# Accede a la documentación: http://localhost:8000/docs

# 6. Iniciar la interfaz gráfica (Streamlit)
streamlit run app_front/app.py
