# Imagen base de Airflow
FROM apache/airflow:2.6.0-python3.9

# Usar el usuario airflow (no root)
USER airflow

# Copiar el archivo de dependencias
COPY requirements.txt /work/requirements.txt

# Instalar los paquetes necesarios
RUN pip install --user --no-cache-dir -r /work/requirements.txt

# Exponer puertos típicos de Airflow (puedes ajustar si necesitas otros)
EXPOSE 8080
