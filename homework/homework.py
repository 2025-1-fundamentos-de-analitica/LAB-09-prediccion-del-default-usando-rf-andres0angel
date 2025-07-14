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
import os
import gzip
import json
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def read_compressed_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="zip")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df[df["EDUCATION"] != 0]
    df = df[df["MARRIAGE"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda val: val if val <= 4 else 4)
    return df


def assemble_pipeline() -> Pipeline:
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    transformer = ColumnTransformer(
        transformers=[("cat", encoder, categorical)],
        remainder="passthrough"
    )
    classifier = RandomForestClassifier(random_state=42)
    return Pipeline([("transform", transformer), ("model", classifier)])


def configure_gridsearch(pipe: Pipeline) -> GridSearchCV:
    search_space = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    return GridSearchCV(
        estimator=pipe,
        param_grid=search_space,
        cv=10,
        scoring="balanced_accuracy",
        refit=True,
        n_jobs=-1,
        verbose=2
    )


def export_model(model, destination: str):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with gzip.open(destination, "wb") as stream:
        pickle.dump(model, stream)


def compute_metrics(label: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics",
        "dataset": label,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_confusion(label: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": label,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


def run_pipeline():
    input_dir = "files/input"
    model_dir = "files/models"
    output_dir = "files/output"

    train_df = preprocess(read_compressed_csv(os.path.join(input_dir, "train_data.csv.zip")))
    test_df = preprocess(read_compressed_csv(os.path.join(input_dir, "test_data.csv.zip")))

    X_train, y_train = train_df.drop("default", axis=1), train_df["default"]
    X_test, y_test = test_df.drop("default", axis=1), test_df["default"]

    pipe = assemble_pipeline()
    search = configure_gridsearch(pipe)
    search.fit(X_train, y_train)

    export_model(search, os.path.join(model_dir, "model.pkl.gz"))

    predictions_train = search.predict(X_train)
    predictions_test = search.predict(X_test)

    results = [
        compute_metrics("train", y_train, predictions_train),
        compute_metrics("test", y_test, predictions_test),
        compute_confusion("train", y_train, predictions_train),
        compute_confusion("test", y_test, predictions_test),
    ]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as file:
        for row in results:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    run_pipeline()
