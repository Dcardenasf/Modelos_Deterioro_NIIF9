# Evaluación del modelo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve,
                           precision_score, recall_score, f1_score,
                           roc_auc_score)
import shap

def evaluar_modelo(model, X_test, y_test, model_name):
    """
    Evalúa un modelo entrenado utilizando múltiples métricas.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        model_name: Nombre del modelo para las visualizaciones
    
    Returns:
        dict: Diccionario con las métricas calculadas
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Métricas de clasificación
    print(f"\nMétricas para {model_name}:")
    print("\nMatriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc
    }

def analizar_importancia_variables(models, X_test, feature_names=None):
    """
    Analiza y visualiza la importancia de las variables para diferentes modelos.
    
    Args:
        models (dict): Diccionario con los modelos entrenados
        X_test: Datos de prueba
        feature_names (list): Lista de nombres de las variables
    """
    if feature_names is None:
        feature_names = X_test.columns
    
    # Importancia de variables para Random Forest
    if 'random_forest' in models:
        rf_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': models['random_forest'].feature_importances_
        })
        rf_importances = rf_importances.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=rf_importances.head(15), x='importance', y='feature')
        plt.title('Top 15 Variables más Importantes - Random Forest')
        plt.xlabel('Importancia')
        plt.ylabel('Variable')
        plt.tight_layout()
        plt.show()
    
    # Análisis SHAP para XGBoost
    if 'xgboost' in models:
        explainer = shap.TreeExplainer(models['xgboost'])
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title('Importancia de Variables Global - SHAP Values (XGBoost)')
        plt.tight_layout()
        plt.show()
        
        # Gráfico de dependencia SHAP para las variables más importantes
        if 'random_forest' in models:
            top_vars = rf_importances['feature'].head(3).values
            for var in top_vars:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(var, shap_values, X_test, show=False)
                plt.title(f'Gráfico de Dependencia SHAP - {var}')
                plt.tight_layout()
                plt.show()

def comparar_modelos(modelos, X_test, y_test):
    """
    Compara el rendimiento de múltiples modelos.
    
    Args:
        modelos (dict): Diccionario con los modelos entrenados
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
    
    Returns:
        DataFrame: Tabla comparativa de métricas
    """
    resultados_modelos = pd.DataFrame({
        'Modelo': list(modelos.keys()),
        'ROC-AUC Test': [roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) 
                        for model in modelos.values()],
        'Precisión': [precision_score(y_test, model.predict(X_test)) 
                     for model in modelos.values()],
        'Recall': [recall_score(y_test, model.predict(X_test)) 
                  for model in modelos.values()],
        'F1-Score': [f1_score(y_test, model.predict(X_test)) 
                    for model in modelos.values()]
    })
    
    # Visualizar comparación de métricas
    plt.figure(figsize=(12, 6))
    resultados_modelos.set_index('Modelo').plot(kind='bar')
    plt.title('Comparación de Métricas por Modelo')
    plt.xlabel('Modelo')
    plt.ylabel('Valor')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Identificar el mejor modelo
    mejor_modelo = resultados_modelos.loc[resultados_modelos['ROC-AUC Test'].idxmax()]
    print("\nMejor modelo basado en ROC-AUC:")
    print(f"Modelo: {mejor_modelo['Modelo']}")
    print(f"ROC-AUC: {mejor_modelo['ROC-AUC Test']:.3f}")
    print(f"F1-Score: {mejor_modelo['F1-Score']:.3f}")
    
    return resultados_modelos