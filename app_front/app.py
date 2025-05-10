import sys
import os
import pandas as pd
import streamlit as st
import requests
# Configurar página y # st.set_page_config(page_title="Predicción de Reingreso", layout="wide")
st.title("🩺 Sistema de Predicción de Reingreso Hospitalario")

# URL del endpoint FastAPI
API_URL = "http://localhost:8000/predict"


# Sección de formulario
st.markdown("## 🧾 Formulario de Evaluación del Paciente")
with st.form("formulario_prediccion"):
    col1, col2, col3 = st.columns(3)

    with col1:
        race = st.selectbox("Raza", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
        gender = st.selectbox("Género", ["Male", "Female"])
        age = st.slider("Edad (años)", 0, 100, 65)
        time_in_hospital = st.number_input("Días en hospital", min_value=1, value=3)
        num_lab_procedures = st.number_input("Procedimientos de laboratorio", min_value=0, value=44)
        num_procedures = st.number_input("Procedimientos", min_value=0, value=0)
        num_medications = st.number_input("Medicamentos", min_value=0, value=13)

    with col2:
        number_outpatient = st.number_input("Visitas ambulatorias", min_value=0, value=0)
        number_emergency = st.number_input("Visitas a urgencias", min_value=0, value=0)
        number_inpatient = st.number_input("Rehospitalizaciones", min_value=0, value=0)
        number_diagnoses = st.slider("Diagnósticos", 0, 20, 9)
        max_glu_serum = st.selectbox("Glucosa en suero máx.", ["None", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("Resultado A1C", ["None", "Norm", ">7", ">8"])

    with col3:
        change = st.selectbox("Cambio de medicamentos", ["No", "Ch"])
        diabetesMed = st.selectbox("Uso de medicamentos para diabetes", ["Yes", "No"])
        admission_type_id = st.selectbox("Tipo de admisión", ["Emergencia", "Electiva", "Recién Nacido", "Centro de Trauma"])
        discharge_disposition_id = st.selectbox("Destino al alta", ["Alta a casa", "Otro"])
        admission_source_id = st.selectbox("Fuente de admisión", ["Referencia", "Emergencia", "Otro"])
        examide = st.selectbox("Uso de examide", ["No", "Steady", "Up", "Down"])
        citoglipton = st.selectbox("Uso de citoglipton", ["No", "Steady", "Up", "Down"])

    submitted = st.form_submit_button("🔍 Predecir")

if submitted:

    payload = {
        "race": race,
        "gender": gender,
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "change": change,
        "diabetesMed": diabetesMed,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "examide": examide,
        "citoglipton": citoglipton
    }

    try:
        
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        resultado = response.json()

        st.success("Predicción realizada correctamente")
        st.metric("Resultado", "Readmitido" if resultado["prediction"] == 1 else "No Readmitido")

        if resultado["probability"] is not None:
            st.metric("Probabilidad de readmisión", f"{resultado['probability'] * 100:.2f}%")
        else:
            st.warning(" El modelo no retornó probabilidad.")

    except requests.exceptions.RequestException as e:
        st.error(f" Error al conectar con la API: {e}")
