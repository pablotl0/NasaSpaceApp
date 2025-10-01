import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Configuración
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================
# Cargar dataset
# ============================
df = pd.read_csv("./datasets/completos/v1.csv")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# Random Forest con GridSearch
# ============================
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

rf_param_grid = {
    "rf__n_estimators": [200, 300, 500],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=2
)

logging.info("Entrenando Random Forest con GridSearch...")
rf_grid.fit(X_train, y_train)

y_pred_rf = rf_grid.predict(X_test)
y_prob_rf = rf_grid.predict_proba(X_test)[:, 1]

# ============================
# XGBoost con GridSearch
# ============================
xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1
    ))
])

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_param_grid = {
    "xgb__n_estimators": [300, 500],
    "xgb__max_depth": [4, 6, 8],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
    "xgb__scale_pos_weight": [scale_pos_weight]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=2
)

logging.info("Entrenando XGBoost con GridSearch...")
xgb_grid.fit(X_train, y_train)

y_pred_xgb = xgb_grid.predict(X_test)
y_prob_xgb = xgb_grid.predict_proba(X_test)[:, 1]

# ============================
# Guardar resumen en TXT
# ============================

RESULTS_FILE = "./resultados/v1.txt"
with open( RESULTS_FILE , "w") as f:
    # RF
    f.write("=== Random Forest ===\n")
    f.write(f"Mejores parámetros RF: {rf_grid.best_params_}\n")
    f.write(f"Mejor ROC-AUC CV RF: {rf_grid.best_score_:.4f}\n\n")
    f.write("Reporte clasificación RF:\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write(f"\nROC-AUC Test RF: {roc_auc_score(y_test, y_prob_rf):.4f}\n")
    f.write(f"Matriz de confusión RF:\n{confusion_matrix(y_test, y_pred_rf)}\n\n")

    # XGB
    f.write("=== XGBoost ===\n")
    f.write(f"Mejores parámetros XGB: {xgb_grid.best_params_}\n")
    f.write(f"Mejor ROC-AUC CV XGB: {xgb_grid.best_score_:.4f}\n\n")
    f.write("Reporte clasificación XGB:\n")
    f.write(classification_report(y_test, y_pred_xgb))
    f.write(f"\nROC-AUC Test XGB: {roc_auc_score(y_test, y_prob_xgb):.4f}\n")
    f.write(f"Matriz de confusión XGB:\n{confusion_matrix(y_test, y_pred_xgb)}\n")

logging.info("Resumen guardado en 'resultados.txt'.")

# ============================
# Guardar todos los resultados a CSV
# ============================
rf_results = pd.DataFrame(rf_grid.cv_results_)
rf_results.to_csv("gridsearch_rf_results.csv", index=False)

xgb_results = pd.DataFrame(xgb_grid.cv_results_)
xgb_results.to_csv("gridsearch_xgb_results.csv", index=False)

logging.info("Resultados completos guardados en CSV (rf y xgb).")

# ============================
# Análisis visual de hiperparámetros
# ============================
def plot_param_performance(results, param_name, title):
    df = pd.DataFrame(results)
    df = df.sort_values(by=f"param_{param_name}")
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        x=f"param_{param_name}",
        y="mean_test_score",
        data=df,
        marker="o"
    )
    plt.title(f"{title} - efecto de {param_name} en ROC-AUC")
    plt.ylabel("ROC-AUC CV")
    plt.xlabel(param_name)
    plt.tight_layout()
    plt.show()

# Ejemplos de gráficas:
plot_param_performance(rf_results, "rf__n_estimators", "Random Forest")
plot_param_performance(rf_results, "rf__max_depth", "Random Forest")
plot_param_performance(xgb_results, "xgb__max_depth", "XGBoost")
plot_param_performance(xgb_results, "xgb__learning_rate", "XGBoost")
