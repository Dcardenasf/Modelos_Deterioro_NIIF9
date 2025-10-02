# Modelos de deterioro
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

def verificar_datos(X, nombre="Dataset"):
    """
    Verifica la presencia de valores faltantes e infinitos en el dataset.
    
    Args:
        X (DataFrame): Dataset a verificar
        nombre (str): Nombre identificativo del dataset
    
    Returns:
        dict: Diccionario con el conteo de valores faltantes e infinitos
    """
    nas = X.isna().sum()
    nas_total = nas.sum()
    inf_count = np.isinf(X.select_dtypes(include=np.number)).sum().sum()
    
    print(f"\nVerificación de {nombre}:")
    if nas_total > 0:
        print("\nColumnas con valores faltantes:")
        print(nas[nas > 0])
    print(f"\nTotal de valores faltantes: {nas_total}")
    print(f"Valores infinitos: {inf_count}")
    
    return {'missing': nas_total, 'infinite': inf_count}

def limpiar_datos(X):
    """
    Limpia el dataset reemplazando valores infinitos y faltantes.
    
    Args:
        X (DataFrame): Dataset a limpiar
    
    Returns:
        DataFrame: Dataset limpio
    """
    X_clean = X.copy()
    # Reemplazar infinitos con NaN
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    # Para cada columna, imputar con la mediana si hay valores faltantes
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            mediana = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(mediana)
    
    return X_clean

def preparar_datos(df, features_numericas, features_categoricas, target='default', test_size=0.2, random_state=42):
    """
    Prepara los datos para el modelamiento, incluyendo división train/test y escalado.
    
    Args:
        df (DataFrame): Dataset completo
        features_numericas (list): Lista de columnas numéricas
        features_categoricas (list): Lista de columnas categóricas
        target (str): Nombre de la columna objetivo
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla aleatoria
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Preparar X e y
    X = df[features_numericas + features_categoricas].copy()
    y = df[target]
    
    # Codificar variables categóricas
    X = pd.get_dummies(X, columns=features_categoricas)
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, 
                                                        stratify=y)
    
    # Limpiar datos
    X_train = limpiar_datos(X_train)
    X_test = limpiar_datos(X_test)
    
    # Escalar datos
    scaler = StandardScaler()
    columnas_numericas = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])
    X_test_scaled[columnas_numericas] = scaler.transform(X_test[columnas_numericas])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def entrenar_modelos(X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa múltiples modelos.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
    
    Returns:
        tuple: (diccionario de modelos, diccionario de resultados)
    """
    models = {
        'logistic': LogisticRegression(class_weight='balanced', max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'xgboost': xgb.XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                                    random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Entrenamiento
        print(f"\nEntrenando {name}...")
        model.fit(X_train, y_train)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Predicciones
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Guardar resultados
        results[name] = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name}:")
        print(f"CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    return models, results

def guardar_modelos(models, directorio='../models'):
    """
    Guarda los modelos entrenados en disco.
    
    Args:
        models (dict): Diccionario con los modelos entrenados
        directorio (str): Ruta donde guardar los modelos
    """
    import os
    os.makedirs(directorio, exist_ok=True)
    
    for name, model in models.items():
        joblib.dump(model, f'{directorio}/{name}_model.pkl')
    
    print(f"\nModelos guardados en el directorio '{directorio}'")