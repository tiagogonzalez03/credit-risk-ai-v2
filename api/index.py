from flask import Flask, request, jsonify
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

modelo = None
dados_cache = None


# =========================
# CARREGAR DADOS
# =========================
def carregar_dataset():
    global dados_cache

    if dados_cache is not None:
        return dados_cache

    # caminho robusto (Vercel-safe)
    base_path = os.path.dirname(__file__)
    file_path = os.path.abspath(
        os.path.join(base_path, '..', 'data', 'SPGlobal_Export_4-14-2026_FinalVersion.csv')
    )

    print("PATH:", file_path)
    print("EXISTS:", os.path.exists(file_path))

    df = pd.read_csv(file_path, encoding='latin-1')

    # =========================
    # CONVERSÃO SEGURA
    # =========================
    def to_float(col):
        return pd.to_numeric(
            col.astype(str).str.replace(',', ''),
            errors='coerce'
        )

    df["Divida"] = to_float(df.iloc[:, 3])
    df["Divida_2023"] = to_float(df.iloc[:, 2])

    df["EBITDA"] = to_float(df.iloc[:, 9])
    df["EBITDA_2023"] = to_float(df.iloc[:, 8])

    # =========================
    # LIMPEZA
    # =========================
    df = df.dropna(subset=["Divida", "EBITDA"])
    df = df[df["EBITDA"] != 0]

    # =========================
    # FEATURES
    # =========================
    df["Alavancagem"] = df["Divida"] / df["EBITDA"]

    df["Crescimento_Divida"] = (
        (df["Divida"] - df["Divida_2023"]) / df["Divida_2023"]
    )

    df["Crescimento_EBITDA"] = (
        (df["EBITDA"] - df["EBITDA_2023"]) / df["EBITDA_2023"]
    )

    df = df.replace([float('inf'), -float('inf')], 0).fillna(0)

    # =========================
    # TARGET
    # =========================
    df["Default"] = (df["Alavancagem"] > 4.5).astype(int)

    print("DATASET SIZE:", len(df))

    dados_cache = df
    return df


# =========================
# TREINAR MODELO
# =========================
def treinar_modelo():
    global modelo

    df = carregar_dataset()

    if len(df) < 10:
        raise Exception("Dataset muito pequeno")

    X = df[[
        "Alavancagem",
        "Crescimento_Divida",
        "Crescimento_EBITDA"
    ]]

    y = df["Default"]

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X, y)


# =========================
# PREVER
# =========================
def prever(dados_empresa):
    global modelo

    if modelo is None:
        treinar_modelo()

    X = [[
        dados_empresa["Alavancagem"],
        dados_empresa["Crescimento_Divida"],
        dados_empresa["Crescimento_EBITDA"]
    ]]

    prob = modelo.predict_proba(X)[0][1]
    return float(prob)


# =========================
# SCORE
# =========================
def gerar_score(prob):
    if prob < 0.05:
        return "AAA"
    elif prob < 0.10:
        return "AA"
    elif prob < 0.20:
        return "A"
    elif prob < 0.30:
        return "BBB"
    elif prob < 0.50:
        return "BB"
    elif prob < 0.70:
        return "B"
    else:
        return "D"


# =========================
# API
# =========================
@app.route('/')
def api():
    empresa_query = request.args.get('empresa', '').lower()
    tipo = request.args.get('tipo', '')

    df = carregar_dataset()

    # TOP RISCO
    if tipo == "top-risk":
        df_sorted = df.sort_values(by="Alavancagem", ascending=False)

        resultado = []

        for _, row in df_sorted.head(10).iterrows():
            resultado.append({
                "Empresa": row.iloc[0],
                "Alavancagem": round(row["Alavancagem"], 2)
            })

        return jsonify(resultado)

    # BUSCA
    if empresa_query:
        resultados = []

        for _, row in df.iterrows():
            nome = str(row.iloc[0]).lower()

            if empresa_query in nome:

                dados_empresa = {
                    "Empresa": row.iloc[0],
                    "Divida_2024": float(row["Divida"]),
                    "EBITDA_2024": float(row["EBITDA"]),
                    "Alavancagem": float(row["Alavancagem"]),
                    "Crescimento_Divida": float(row["Crescimento_Divida"]),
                    "Crescimento_EBITDA": float(row["Crescimento_EBITDA"])
                }

                prob = prever(dados_empresa)
                score = gerar_score(prob)

                dados_empresa["Prob_Default"] = round(prob, 3)
                dados_empresa["Score"] = score

                resultados.append(dados_empresa)

        return jsonify(resultados[:10])

    return jsonify({"status": "ok"})


# =========================
# VERCEL EXPORT (ESSENCIAL)
# =========================
app = app
