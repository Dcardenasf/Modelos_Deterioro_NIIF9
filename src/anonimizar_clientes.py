import pandas as pd
import hashlib

def clean_id(id_value):
    # Convertir a string
    id_str = str(id_value)
    # Eliminar espacios y puntos
    id_str = id_str.strip().replace('.', '').replace(' ', '')
    # Eliminar ceros a la izquierda
    id_str = id_str.lstrip('0')
    return id_str

def hash_id(id_value):
    # Limpiar el ID primero
    id_str = clean_id(id_value)
    # Crear el hash
    return hashlib.sha256(id_str.encode()).hexdigest()

def is_hash(value):
    # Verifica si una cadena parece ser un hash SHA-256 (64 caracteres hexadecimales)
    try:
        return len(value) == 64 and all(c in '0123456789abcdef' for c in value.lower())
    except:
        return False

def main():
    # Rutas de los archivos
    datos_abiertos_path = "data/raw/datos_abiertos.csv"
    facturas_path = "data/raw/Historico_Facturas.csv"
    cierres_path = "data/raw/Historico_Cierres.csv"
    clientes_output_path = "data/raw/Historico_Clientes.csv"

    # Leer los archivos
    print("Leyendo archivos...")
    df_abiertos = pd.read_csv(datos_abiertos_path, low_memory=False)
    df_facturas = pd.read_csv(facturas_path)
    df_cierres = pd.read_csv(cierres_path, sep=';')

    # Aplicar hash a los IDs de datos_abiertos
    print("\nProcesando identificadores...")
    df_abiertos['id_hash'] = df_abiertos['numero_identificacion'].apply(hash_id)

    # Obtener valores únicos de NITs de ambos archivos
    ids_facturas = set(df_facturas['NIT'].unique())
    ids_cierres = set(df_cierres['NIT'].unique())

    # Unir los conjuntos de IDs
    ids_totales = ids_facturas.union(ids_cierres)

    print(f"\nResumen de IDs:")
    print(f"IDs únicos en Facturas: {len(ids_facturas)}")
    print(f"IDs únicos en Cierres: {len(ids_cierres)}")
    print(f"Total IDs únicos combinados: {len(ids_totales)}")

    # Filtrar datos_abiertos para mantener solo los que están en alguno de los dos archivos
    print("\nCruzando información...")
    df_clientes = df_abiertos[df_abiertos['id_hash'].isin(ids_totales)].copy()

    # Eliminar la columna de número de identificación
    print("Eliminando columna de identificación original...")
    df_clientes = df_clientes.drop(columns=['numero_identificacion'])

    # Agregar columnas de presencia en cada archivo
    df_clientes['presente_en_facturas'] = df_clientes['id_hash'].isin(ids_facturas)
    df_clientes['presente_en_cierres'] = df_clientes['id_hash'].isin(ids_cierres)

    # Guardar el nuevo archivo
    print("Guardando archivo resultado...")
    df_clientes.to_csv(clientes_output_path, index=False)

    print(f"\nResumen del proceso:")
    print(f"Total registros en datos_abiertos: {len(df_abiertos)}")
    print(f"Total registros en archivo final: {len(df_clientes)}")
    print("\nDistribución de presencia en archivos:")
    distribucion = df_clientes[['presente_en_facturas', 'presente_en_cierres']].value_counts()
    for idx, count in distribucion.items():
        print(f"En Facturas: {idx[0]}, En Cierres: {idx[1]} -> {count} registros")
    print(f"\nArchivo guardado en: {clientes_output_path}")

if __name__ == "__main__":
    main()
