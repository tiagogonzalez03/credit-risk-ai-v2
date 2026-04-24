import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

modelo = None

# =========================
# CARREGAR DATASET
# =========================
def carregar_dataset():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(
        base_path,
        '..',
        'data',
        'SPGlobal_Export_4-14-2026_FinalVersion.csv'
    )

    print("CSV PATH:", file_path)

    if not os.path.exists(file_path):
        print("❌ CSV NÃO ENCONTRADO")
        return None

    df = pd.read_csv(file_path, encoding='latin-1')

    # converter colunas com segurança
    df["Divida"] = pd.to_numeric(df.iloc[:, 3].astype(str).str.replace(',', ''), errors='coerce')
    df["EBITDA"] = pd.to_numeric(df.iloc[:, 9].astype(str).str.replace(',', ''), errors='coerce')

    # remover apenas linhas inválidas (mais seguro que dropna total)
    df = df[(df["EBITDA"] > 0) & (df["Divida"] > 0)]

    # evitar dataset vazio
    if df.empty:
        print("❌ DATASET VAZIO APÓS LIMPEZA")
        return None

    # feature
    df["Alavancagem"] = df["Divida"] / df["EBITDA"]

    # limitar outliers (importante)
    df["Alavancagem"] = df["Alavancagem"].clip(upper=20)

    # target (proxy)
    df["Default"] = (df["Alavancagem"] > 4.5).astype(int)

    return df[["Alavancagem", "Default"]]


# =========================
# TREINAR MODELO
# =========================
def treinar_modelo():
    global modelo

    df = carregar_dataset()

    if df is None:
        print("❌ ERRO AO CARREGAR DATASET")
        modelo = None
        return

    X = df[["Alavancagem"]]
    y = df["Default"]

    modelo = LogisticRegression()
    modelo.fit(X, y)

    print("✅ MODELO TREINADO")


# =========================
# PREVISÃO
# =========================
def prever(alavancagem):
    global modelo

    try:
        if modelo is None:
            treinar_modelo()

        if modelo is None:
            return 0.12  # fallback seguro

        prob = modelo.predict_proba([[alavancagem]])[0][1]
        return float(prob)

    except Exception as e:
        print("❌ ERRO NA PREVISÃO:", e)
        return 0.12
