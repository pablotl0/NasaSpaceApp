import pandas as pd
import numpy as np
import logging
from typing import Optional
import os

# ============================
# Configuración de logging
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================
# Funciones de preprocesamiento
# ============================

def preprocess_kepler(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el dataset KOI (Kepler)."""
    logging.info("Procesando dataset Kepler (KOI)...")
    return pd.DataFrame({
        "orbital_period": df["koi_period"],
        "impact_parameter": df["koi_impact"],
        "transit_duration": df["koi_duration"],
        "transit_depth": df["koi_depth"],  # ppm
        "planet_radius": df["koi_prad"],
        "planet_star_radius_ratio": df["koi_ror"],
        "stellar_teff": df["koi_steff"],
        "stellar_logg": df["koi_slogg"],
        "stellar_radius": df["koi_srad"],
        "stellar_mass": df["koi_smass"],
        "stellar_density": df["koi_srho"],
        "stellar_magnitude": df["koi_kepmag"],
        "label": df["koi_disposition"].str.upper().apply(
            lambda x: 1 if "CONFIRMED" in x else 0
        )
    })


def preprocess_tess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el dataset TOI (TESS)."""
    logging.info("Procesando dataset TESS (TOI)...")
    return pd.DataFrame({
        "orbital_period": df["pl_orbper"],
        "impact_parameter": np.nan,  # No disponible
        "transit_duration": df["pl_trandurh"],
        "transit_depth": df["pl_trandep"],  # ppm
        "planet_radius": df["pl_rade"],
        "planet_star_radius_ratio": np.nan,
        "stellar_teff": df["st_teff"],
        "stellar_logg": df["st_logg"],
        "stellar_radius": df["st_rad"],
        "stellar_mass": np.nan,
        "stellar_density": np.nan,
        "stellar_magnitude": df["st_tmag"],
        "label": df["tfopwg_disp"].str.upper().apply(
            lambda x: 1 if "CONFIRMED" in x or "KP" in x else 0
        )
    })


def preprocess_k2(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el dataset K2."""
    logging.info("Procesando dataset K2...")
    return pd.DataFrame({
        "orbital_period": df["pl_orbper"],
        "impact_parameter": df["pl_imppar"],
        "transit_duration": df["pl_trandur"],
        "transit_depth": df["pl_trandep"] * 10000,  # % → ppm
        "planet_radius": df["pl_rade"],
        "planet_star_radius_ratio": df["pl_ratror"],
        "stellar_teff": df["st_teff"],
        "stellar_logg": df["st_logg"],
        "stellar_radius": df["st_rad"],
        "stellar_mass": df["st_mass"],
        "stellar_density": df["st_dens"],
        "stellar_magnitude": np.nan,
        "label": df["disposition"].str.upper().apply(
            lambda x: 1 if "CONFIRMED" in x else 0
        )
    })


def clean_and_merge(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Combina datasets, limpia valores y prepara dataset final."""
    logging.info("Uniendo datasets...")

    df = pd.concat(datasets, ignore_index=True)

    # Remover filas sin etiqueta
    df = df.dropna(subset=["label"])

    # Reemplazar infinitos
    df = df.replace([np.inf, -np.inf], np.nan)

    # Imputación de valores faltantes con mediana por columna
    for col in df.columns:
        if col != "label":
            median_val: Optional[float] = df[col].median()
            df[col] = df[col].fillna(median_val)

    logging.info(f"Dataset final: {df.shape[0]} filas, {df.shape[1]} columnas")
    logging.info(f"Balance de clases:\n{df['label'].value_counts()}")
    return df


# ============================
# Pipeline principal
# ============================

def build_training_dataset(
    kepler_path: str,
    tess_path: str,
    k2_path: str,
    output_path: str = "./datasets/completos/v1.csv"
) -> pd.DataFrame:
    """Construye el dataset de entrenamiento unificado a partir de Kepler, TESS y K2."""

    # Verificar que los archivos existen
    for name, path in [("Kepler", kepler_path), ("TESS", tess_path), ("K2", k2_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo {name} no encontrado en: {os.path.abspath(path)}")
    # Cargar
    logging.info("Cargando archivos...")
    koi_raw = pd.read_csv(kepler_path, comment='#')
    toi_raw = pd.read_csv(tess_path, comment='#')
    k2_raw = pd.read_csv(k2_path, comment='#')

    # Preprocesar cada misión
    koi_df = preprocess_kepler(koi_raw)
    toi_df = preprocess_tess(toi_raw)
    k2_df = preprocess_k2(k2_raw)

    # Unir y limpiar
    final_df = clean_and_merge([koi_df, toi_df, k2_df])

    # Guardar
    final_df.to_csv(output_path, index=False)
    logging.info(f"Dataset guardado en {output_path}")

    return final_df


# ============================
# Ejecución directa
# ============================
if __name__ == "__main__":
    dataset = build_training_dataset(
        "./datasets/preprocesados/v1/KOI1.csv",
        "./datasets/preprocesados/v1/TOI1.csv",
        "./datasets/preprocesados/v1/K21.csv"
    )
