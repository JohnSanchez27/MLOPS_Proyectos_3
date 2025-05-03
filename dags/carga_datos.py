import os
import requests

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
