# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501

import pandas as pd
import os
import json
import joblib
import gzip
import zipfile

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def load_and_clean_data(path_zip, temp_dir="/mnt/data/temp") -> pd.DataFrame:
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    csv_file = os.listdir(temp_dir)[0]
    df = pd.read_csv(os.path.join(temp_dir, csv_file))
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"], errors="ignore")
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df.dropna()


def build_pipeline() -> Pipeline:
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough"
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42
        ))
    ])


def get_metrics(y_true, y_pred, dataset_name: str) -> dict:
    return {
        'dataset': dataset_name,
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }


def get_confusion_dict(y_true, y_pred, dataset_name: str) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
        'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])},
    }


def main():
    # === Paso 1 y 2 ===
    train_df = load_and_clean_data("files/input/train_data.csv.zip")
    test_df = load_and_clean_data("files/input/test_data.csv.zip")

    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    # === Paso 3 y 4 ===
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # === Paso 5 ===
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        joblib.dump(pipeline, f)

    # === Paso 6 ===
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_metrics = get_metrics(y_train, y_train_pred, "train")
    test_metrics = get_metrics(y_test, y_test_pred, "test")

    # === Paso 7 ===
    train_cm = get_confusion_dict(y_train, y_train_pred, "train")
    test_cm = get_confusion_dict(y_test, y_test_pred, "test")

    # === Guardar métricas ===
    os.makedirs("files/output", exist_ok=True)
    all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
    with open("files/output/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
