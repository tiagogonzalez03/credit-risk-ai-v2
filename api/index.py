from flask import Flask, request, jsonify
from .model import prever
import csv
import os
import unicodedata

app = Flask(__name__)

dados_cache = None

# =========================
# NORMALIZAR TEXTO
# =========================
def limpar_texto(texto):
    return unicodedata.normalize('NFKD', texto)\
        .encode('ascii', 'ignore')\
        .decode('utf-8')\
        .lower().strip()

# =========================
# CONVERSÃO SEGURA
# =========================
def to_float(value):
    try:
        return float(str(value).replace(',', '').strip())
    except:
        return 0.0

# =========================
# SCORE DE CRÉDITO
# =========================
def gerar_score(prob):
    if prob < 0.03:
        return "AAA"
    elif prob < 0.07:
        return "AA"
    elif prob < 0.12:
        return "A"
    elif prob < 0.2:
        return "BBB"
    elif prob < 0.3:
        return "BB"
    else:
        return "B"

# =========================
# INTERPRETAÇÃO
# =========================
def gerar_analise(d):
    if d["Alavancagem"] is None:
        return "Dados insuficientes."

    if d["Alavancagem"] < 1:
        return "Baixo nível de dívida."
    elif d["Alavancagem"] < 3:
        return "Estrutura de capital equilibrada."
    elif d["Alavancagem"] < 5:
        return "Atenção ao nível de endividamento."
    else:
        return "Alto risco financeiro."

# =========================
# CARREGAR DADOS
# =========================
def carregar_dados():
    global dados_cache

    if dados_cache is not None:
        return dados_cache

    file_path = 'data/SPGlobal_Export_4-14-2026_FinalVersion.csv'

    dados = []

    if not os.path.exists(file_path):
        return []

    with open(file_path, newline='', encoding='latin-1') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if not row or len(row) < 10:
                continue

            try:
                empresa = row[0].strip()

                divida_2024 = to_float(row[3])
                divida_2023 = to_float(row[2])

                ebitda_2024 = to_float(row[9])
                ebitda_2023 = to_float(row[8])

                alavancagem = (
                    divida_2024 / ebitda_2024
                    if ebitda_2024 != 0 else None
                )

                crescimento_divida = (
                    (divida_2024 - divida_2023) / divida_2023
                    if divida_2023 != 0 else 0
                )

                crescimento_ebitda = (
                    (ebitda_2024 - ebitda_2023) / ebitda_2023
                    if ebitda_2023 != 0 else 0
                )

                dados.append({
                    "Empresa": empresa,
                    "Divida_2024": divida_2024,
                    "EBITDA_2024": ebitda_2024,
                    "Alavancagem": round(alavancagem, 2) if alavancagem else None,
                    "Crescimento_Divida": crescimento_divida,
                    "Crescimento_EBITDA": crescimento_ebitda
                })

            except:
                continue

    dados_cache = dados
    return dados_cache

# =========================
# API
# =========================
@app.route('/')
def api():
    tipo = request.args.get('tipo', '')
    empresa_query = request.args.get('empresa', '')

    dados = carregar_dados()

    # 🔥 TOP RISCO
    if tipo == "top-risk":
        ordenado = sorted(
            dados,
            key=lambda x: x["Alavancagem"] if x["Alavancagem"] is not None else -1,
            reverse=True
        )
        return jsonify(ordenado[:10])

    # 🔎 BUSCA
    if empresa_query:
        query = limpar_texto(empresa_query)
        resultados = []

        for item in dados:
            nome = limpar_texto(item["Empresa"])

            if query in nome:

                # ✅ CORREÇÃO AQUI
                if item["Alavancagem"] is not None:
                    try:     prob = prever(item["Alavancagem"]) except:     prob = 0.12
                else:
                    try:     prob = prever(item["Alavancagem"]) except:     prob = 0.12

                score = gerar_score(prob)

                item["Prob_Default"] = round(prob, 3)
                item["Score"] = score
                item["Analise"] = gerar_analise(item)

                resultados.append(item)

        return jsonify(resultados[:10])

    return jsonify({"status": "ok"})
