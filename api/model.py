import pandas as pd
from sklearn.linear_model import LogisticRegression

modelo = None

def carregar_dataset():
    df = pd.read_csv('data/SPGlobal_Export_4-14-2026_FinalVersion.csv', encoding='latin-1')

    # remover lixo
    df = df.dropna()

    # converter colunas
    df["Divida"] = df.iloc[:, 3].replace(',', '', regex=True).astype(float)
    df["EBITDA"] = df.iloc[:, 9].replace(',', '', regex=True).astype(float)

    # criar feature
    df["Alavancagem"] = df["Divida"] / df["EBITDA"]

    # variável alvo (proxy)
    df["Default"] = (df["Alavancagem"] > 4.5).astype(int)

    return df[["Alavancagem", "Default"]]


def treinar_modelo():
    global modelo

    df = carregar_dataset()

    X = df[["Alavancagem"]]
    y = df["Default"]

    modelo = LogisticRegression()
    modelo.fit(X, y)


def prever(alavancagem):
    global modelo

    if modelo is None:
        treinar_modelo()

    prob = modelo.predict_proba([[alavancagem]])[0][1]
    return float(prob)
