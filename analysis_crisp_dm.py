import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar src al path para importar funciones helper
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from procesamiento_datos import revision_inicial, convertir_fecha, convertir_numerico

# Configuración de visualización
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

DATA_DIR = r'data/raw'
PLOTS_DIR = r'plots'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def load_data():
    """Carga todos los archivos CSV en la carpeta data/raw."""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    datasets = {}
    print(f"Archivos encontrados en {DATA_DIR}: {files}\n")
    
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        try:
            # Intentar cargar con encoding utf-8, si falla probar latin-1
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            
            datasets[file] = df
            print(f"--> Cargado: {file} | Shape: {df.shape}")
        except Exception as e:
            print(f"Error cargando {file}: {e}")
            
    return datasets

def save_plot(fig, name):
    """Guarda la figura actual en la carpeta plots."""
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f"[GRÁFICO] Guardado: {path}")
    plt.close(fig)

def plot_distributions(name, df, sample_size=10000):
    """Genera histogramas para variables numéricas clave con muestreo para datasets grandes."""
    print(f"   [PLOT] Generando distribuciones para {name}...")
    try:
        # Aplicar muestreo si el dataset es muy grande
        if len(df) > sample_size:
            print(f"   [PLOT] Dataset grande ({len(df)} filas), usando muestra de {sample_size} filas")
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
            
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        # Filtrar columnas con pocos valores únicos (posibles categóricas codificadas) o IDs
        cols_to_plot = [c for c in numeric_cols if df_sample[c].nunique() > 10 and 'ID' not in c.upper() and 'NIT' not in c.upper()]
        
        if not cols_to_plot:
            print(f"   [PLOT] No hay columnas numéricas adecuadas para {name}")
            return

        # Limitar a 6 variables para no saturar
        cols_to_plot = cols_to_plot[:6]
        
        fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 4 * len(cols_to_plot)))
        if len(cols_to_plot) == 1:
            axes = [axes]
            
        for i, col in enumerate(cols_to_plot):
            sns.histplot(df_sample[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribución de {col} - {name}')
            
        plt.tight_layout()
        save_plot(fig, f"dist_{name.replace('.csv', '.png')}")
    except Exception as e:
        print(f"   [ERROR] Falló plot_distributions para {name}: {e}")

def plot_categorical(name, df, sample_size=10000):
    """Genera gráficos de barras para variables categóricas clave con muestreo para datasets grandes."""
    print(f"   [PLOT] Generando categóricas para {name}...")
    try:
        # Aplicar muestreo si el dataset es muy grande
        if len(df) > sample_size:
            print(f"   [PLOT] Dataset grande ({len(df)} filas), usando muestra de {sample_size} filas")
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
            
        cat_cols = df_sample.select_dtypes(include=['object', 'category']).columns
        # Filtrar columnas con demasiados valores únicos (como IDs, nombres, fechas)
        cols_to_plot = [c for c in cat_cols if df_sample[c].nunique() < 20 and df_sample[c].nunique() > 1]
        
        if not cols_to_plot:
            print(f"   [PLOT] No hay columnas categóricas adecuadas para {name}")
            return

        # Limitar a 4 variables
        cols_to_plot = cols_to_plot[:4]
        
        fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 5 * len(cols_to_plot)))
        if len(cols_to_plot) == 1:
            axes = [axes]
            
        for i, col in enumerate(cols_to_plot):
            sns.countplot(y=df_sample[col], order=df_sample[col].value_counts().index, ax=axes[i])
            axes[i].set_title(f'Frecuencia de {col} - {name}')
            
        plt.tight_layout()
        save_plot(fig, f"cat_{name.replace('.csv', '.png')}")
    except Exception as e:
        print(f"   [ERROR] Falló plot_categorical para {name}: {e}")

def process_dates(name, df):
    """Aplica conversión de fechas según el dataset."""
    print(f"   [PROCESO] Convirtiendo fechas para {name}...")
    try:
        if 'Historico_Cierres' in name:
            if 'Fecha de Vencimiento' in df.columns:
                df = convertir_fecha(df, ['Fecha de Vencimiento'])
                print(f"   [OK] Convertida columna 'Fecha de Vencimiento'")
        
        elif 'Informacion_Clientes' in name:
            if 'FECHA_CORTE' in df.columns:
                df = convertir_fecha(df, ['FECHA_CORTE'])
                print(f"   [OK] Convertida columna 'FECHA_CORTE'")
        
        elif 'Variables_Macro' in name:
            # Buscar columnas que parezcan fechas
            date_cols = [col for col in df.columns if 'FECHA' in col.upper() or 'DATE' in col.upper()]
            if date_cols:
                df = convertir_fecha(df, date_cols)
                print(f"   [OK] Convertidas columnas: {date_cols}")
                
    except Exception as e:
        print(f"   [WARNING] Error en conversión de fechas: {e}")
    
    return df

def explore_dataset(name, df):
    """Realiza exploración inicial usando revision_inicial del módulo helper."""
    print(f"\n{'='*80}")
    print(f"EXPLORACIÓN DE: {name}")
    print(f"{'='*80}\n")
    
    # 1. Procesar fechas
    df = process_dates(name, df)
    
    # 2. Usar revision_inicial del módulo helper
    revision_inicial(df)
    
    # 3. Generar Gráficos
    plot_distributions(name, df)
    plot_categorical(name, df)

def analyze_correlations(name, df, sample_size=10000):
    """Genera matriz de correlación para variables numéricas con muestreo para datasets grandes."""
    # Aplicar muestreo si el dataset es muy grande
    if len(df) > sample_size:
        print(f"   [CORR] Dataset grande ({len(df)} filas), usando muestra de {sample_size} filas")
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
        
    numeric_df = df_sample.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        fig = plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Matriz de Correlación - {name}')
        plt.tight_layout()
        save_plot(fig, f"corr_{name.replace('.csv', '.png')}")
    else:
        print(f"\nNo hay suficientes variables numéricas para correlación en {name}")

def main():
    print("INICIANDO ANÁLISIS CRISP-DM - ETAPAS 1 & 2")
    
    # 1. Carga de Datos
    datasets = load_data()
    
    # 2. Exploración y Calidad
    summary_list = []
    
    for name, df in datasets.items():
        explore_dataset(name, df)
        analyze_correlations(name, df)
        
        # Guardar resumen para tabla final
        summary_list.append({
            'Dataset': name,
            'Filas': df.shape[0],
            'Columnas': df.shape[1],
            'Nulos Totales': df.isnull().sum().sum(),
            'Duplicados': df.duplicated().sum()
        })
        
    # 3. Resumen Final
    print(f"\n{'='*80}")
    print("RESUMEN DE DATASETS")
    print(f"{'='*80}")
    summary_df = pd.DataFrame(summary_list)
    print(summary_df)

if __name__ == "__main__":
    main()
