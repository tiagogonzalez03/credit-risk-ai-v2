from fastapi import FastAPI
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

app = FastAPI()

modelo = None
dados_cache = None

def carregar_dataset():
    global dados_cache

    if dados_cache is not None:
        return dados_cache

    base_path = os.path.dirname(__file__)
    file_path = os.path.abspath(
        os.path.join(base_path, '..', 'data', 'SPGlobal_Export_4-14-2026_FinalVersion.csv')
    )

    df = pd.read_csv(file_path, encoding='latin-1')

    def to_float(col):
        return pd.to_numeric(col.astype(str).str.replace(',', ''), errors='coerce')

    df["Divida"] = to_float(df.iloc[:, 3])
    df["EBITDA"] = to_float(df.iloc[:, 9])

    df = df.dropna(subset=["Divida", "EBITDA"])
    df = df[df["EBITDA"] != 0]

    df["Alavancagem"] = df["Divida"] / df["EBITDA"]
    df["Default"] = (df["Alavancagem"] > 4.5).astype(int)

    dados_cache = df
    return df

def treinar():
    global modelo
    df = carregar_dataset()
    X = df[["Alavancagem"]]
    y = df["Default"]

    modelo = LogisticRegression()
    modelo.fit(X, y)

@app.get("/api")
def api(empresa: str = ""):
    df = carregar_dataset()

    if empresa:
        resultados = df[df.iloc[:, 0].str.lower().str.contains(empresa.lower())]

        return resultados.head(10).to_dict(orient="records")

    return {"status": "ok"}
