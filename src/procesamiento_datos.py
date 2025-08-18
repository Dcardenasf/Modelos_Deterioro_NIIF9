# Limpieza y transformación de datos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def boxplot_numericos(df, outliers=True):
    """
    Genera gráficos de caja para variables numéricas en el DataFrame.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    sns.set(style="whitegrid")

    if outliers == True:
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot de {col} con outliers")
            plt.show()
    else:

        df_no_outliers = df.copy()
        for col in num_cols:
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]

        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df_no_outliers[col])
            plt.title(f"Boxplot de {col} sin outliers")
            plt.show()

def hist_numericos(df, outliers=True):
    """
    Genera histogramas para variables numéricas en el DataFrame.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    sns.set(style="whitegrid")

    if outliers == True:
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Histograma de {col} con outliers")
            plt.show()
    else:
        df_no_outliers = df.copy()
        for col in num_cols:
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]    

        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.histplot(df_no_outliers[col], kde=True)
            plt.title(f"Histograma de {col} sin outliers")
            plt.show()

print("Función de procesamiento de datos cargadas correctamente.")