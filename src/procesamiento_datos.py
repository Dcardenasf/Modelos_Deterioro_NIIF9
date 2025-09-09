# Limpieza y transformación de datos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import datetime as dt
from pandas.tseries.offsets import MonthEnd
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
#pd.set_option('display.max_columns', None)

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
    print("="*60)
    display(df.head())
    print("\nInformación del DataFrame:")
    print("="*60)
    print(df.info())
    try:
        print("\nDescripción datos categoricos del DataFrame:")
        print("="*60)
        print(df.describe(include=[object]))
    except ValueError:
        print("No hay datos categóricos para describir.")
    try:
        print("\nDescripción datos cuantitativos del DataFrame:")
        print("="*60)
        print(df.describe(include=[np.number]))
    except ValueError:
        print("No hay datos cuantitativos para describir.")
    print("\nValores nulos por columna:")
    print("="*60)
    print(df.isnull().sum())
    print("\nValores duplicados:")
    print("="*60)
    print(df.duplicated().sum())
    print("\n" + "="*60)
    print("Fin de revisión inicial para este DataFrame")
    print("="*60 + "\n")
    

def boxplot_numericos(df, outliers=True):
    """
    Genera gráficos de caja para variables numéricas en el DataFrame.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    sns.set(style="whitegrid")

    if outliers == True:
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            if len(df[df[col] > 0]) > 0:  # Si hay valores mayores a 0
                sns.boxplot(x=df[df[col] > 0][col])
                plt.title(f"Boxplot de {col} con outliers (valores > 0)")
            else:
                sns.boxplot(x=df[col])
                plt.title(f"Boxplot de {col} con outliers")
            plt.show()
    else:
        df_no_outliers = df.copy()
        for col in num_cols:
            # Solo procesar si hay variación en los datos
            if df_no_outliers[col].std() > 0:
                Q1 = df_no_outliers[col].quantile(0.25)
                Q3 = df_no_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Solo aplicar si hay dispersión en los datos
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & 
                                                   (df_no_outliers[col] <= upper_bound)]

            plt.figure(figsize=(10, 5))
            if len(df_no_outliers[df_no_outliers[col] > 0]) > 0:  # Si hay valores mayores a 0
                sns.boxplot(x=df_no_outliers[df_no_outliers[col] > 0][col])
                plt.title(f"Boxplot de {col} sin outliers (valores > 0)")
            else:
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
            # Solo procesar si hay variación en los datos
            if df_no_outliers[col].std() > 0:
                Q1 = df_no_outliers[col].quantile(0.25)
                Q3 = df_no_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Solo aplicar si hay dispersión en los datos
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & 
                                                   (df_no_outliers[col] <= upper_bound)]
            
            plt.figure(figsize=(10, 5))
            if len(df_no_outliers[df_no_outliers[col] > 0]) > 0:  # Si hay valores mayores a 0
                sns.histplot(data=df_no_outliers[df_no_outliers[col] > 0], x=col, kde=True)
                plt.title(f"Histograma de {col} sin outliers (valores > 0)")
            else:
                sns.histplot(data=df_no_outliers, x=col, kde=True)
                plt.title(f"Histograma de {col} sin outliers")
            plt.show()

def visualizar_distribucion(df, outliers=True):
    """
    Genera visualizaciones combinadas (boxplot e histograma) para variables numéricas en el DataFrame.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    sns.set(style="whitegrid")

    if outliers == True:
        for col in num_cols:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            if len(df[df[col] > 0]) > 0:  # Si hay valores mayores a 0
                # Boxplot
                sns.boxplot(x=df[df[col] > 0][col], ax=ax1)
                ax1.set_title(f"Boxplot de {col}\n(valores > 0)")
                # Histograma
                sns.histplot(data=df[df[col] > 0], x=col, kde=True, ax=ax2)
                ax2.set_title(f"Histograma de {col}\n(valores > 0)")
            else:
                # Boxplot
                sns.boxplot(x=df[col], ax=ax1)
                ax1.set_title(f"Boxplot de {col}")
                # Histograma
                sns.histplot(data=df, x=col, kde=True, ax=ax2)
                ax2.set_title(f"Histograma de {col}")
            
            plt.tight_layout()
            plt.show()
    else:
        df_no_outliers = df.copy()
        for col in num_cols:
            # Solo procesar si hay variación en los datos
            if df_no_outliers[col].std() > 0:
                Q1 = df_no_outliers[col].quantile(0.25)
                Q3 = df_no_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Solo aplicar si hay dispersión en los datos
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & 
                                                   (df_no_outliers[col] <= upper_bound)]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            if len(df_no_outliers[df_no_outliers[col] > 0]) > 0:  # Si hay valores mayores a 0
                # Boxplot
                sns.boxplot(x=df_no_outliers[df_no_outliers[col] > 0][col], ax=ax1)
                ax1.set_title(f"Boxplot de {col} sin outliers\n(valores > 0)")
                # Histograma
                sns.histplot(data=df_no_outliers[df_no_outliers[col] > 0], x=col, kde=True, ax=ax2)
                ax2.set_title(f"Histograma de {col} sin outliers\n(valores > 0)")
            else:
                # Boxplot
                sns.boxplot(x=df_no_outliers[col], ax=ax1)
                ax1.set_title(f"Boxplot de {col} sin outliers")
                # Histograma
                sns.histplot(data=df_no_outliers, x=col, kde=True, ax=ax2)
                ax2.set_title(f"Histograma de {col} sin outliers")
            
            plt.tight_layout()
            plt.show()

def visualizar_categoricas (df,columna):
    """
    Genera gráficos de barras para variables categóricas en el DataFrame.
    df = DataFrame
    columna = lista de columnas categóricas
    """
    cat_cols = columna
    sns.set(style="whitegrid")

    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Gráfico de barras de {col}")
        plt.xlabel("Conteo")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

print("Función de procesamiento de datos cargadas correctamente.")