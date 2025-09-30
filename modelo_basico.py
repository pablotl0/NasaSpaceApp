import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
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
# Cargar dataset ya unificado
# ============================
df = pd.read_csv("./datasets/completos/v1.csv")

# Separar features y labels
X = df.drop(columns=["label"])
y = df["label"]

# ============================
# Train/test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# Escalado de features
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# Modelo 1: Random Forest
# ============================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

logging.info("Resultados Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# ============================
# Modelo 2: XGBoost
# ============================
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),  # balance clases
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

logging.info("Resultados XGBoost:")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

# ============================
# Matriz de confusión
# ============================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Random Forest - Matriz de Confusión")
ax[0].set_xlabel("Predicho")
ax[0].set_ylabel("Real")

sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("XGBoost - Matriz de Confusión")
ax[1].set_xlabel("Predicho")
ax[1].set_ylabel("Real")

plt.tight_layout()
plt.show()

# ============================
# Feature Importance
# ============================

# Random Forest
importances_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances_rf.plot(kind="bar", figsize=(12, 4), title="Random Forest - Feature Importance")
plt.show()

# XGBoost
importances_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances_xgb.plot(kind="bar", figsize=(12, 4), title="XGBoost - Feature Importance")
plt.show()
