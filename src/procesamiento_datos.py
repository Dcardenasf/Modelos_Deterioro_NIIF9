# Limpieza y transformación de datos
import pandas as pd
import numpy as np
import os
import re

def cargar_datos(ruta_archivo, sep=','):
    """
    Carga los datos desde un archivo CSV.
    """
    try:
        if ruta_archivo.endswith('.csv'):
            return pd.read_csv(ruta_archivo,sep=sep, encoding='utf-8', low_memory=False)
        elif ruta_archivo.endswith('.xlsx'):
            return pd.read_excel(ruta_archivo)
        else:
            raise ValueError("Formato de archivo no soportado. Use .csv o .xlsx")
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {ruta_archivo} no se encuentra.")
    except Exception as e:
        raise Exception(f"Error al cargar el archivo: {e}")

def revision_inicial(df):
    """
    Realiza una revisión inicial del DataFrame.
    """
    print("Primeras filas del DataFrame:")
    display(df.head())
    print("\nInformación del DataFrame:")
    print(df.info())
    try:
        print("\nDescripción datos categoricos del DataFrame:")
        print(df.describe(include=[object]))
    except ValueError:
        print("No hay datos categóricos para describir.")
    try:
        print("\nDescripción datos cuantitativos del DataFrame:")
        print(df.describe(include=[np.number]))
    except ValueError:
        print("No hay datos cuantitativos para describir.")
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("\nValores duplicados:")
    print(df.duplicated().sum())


print("Función de procesamiento de datos cargadas correctamente.")