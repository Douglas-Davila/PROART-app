import json
import os
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "proart_config.json")
CSV_PATH = os.path.join(BASE_DIR, "respostas_proart.csv")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

ITENS_ORDENADOS = []
for escala_info in CONFIG["escalas"].values():
    for fator_info in escala_info["fatores"].values():
        for cod_item in fator_info["itens"].keys():
            ITENS_ORDENADOS.append(cod_item)

CAMPOS_DEMO = [
    "idade",
    "sexo",
    "escolaridade",
    "estado_civil",
    "cargo_atual",
    "lotacao",
    "tempo_instituicao_meses",
    "tempo_cargo_meses",
    "tipo_contrato",
    "problemas_saude_ultimo_ano",
]
COLUNAS_CSV = ["respondent_id"] + CAMPOS_DEMO + ITENS_ORDENADOS


def gerar_likert_centrado(media_alvo: float, dispersao: float = 0.9) -> int:
    val = int(round(random.gauss(media_alvo, dispersao)))
    return max(1, min(5, val))


def gerar_dados(n_respondentes: int = 30, seed: int = 42):
    random.seed(seed)

    # 3 lotações com 10 respondentes cada -> total 30 (ideal para testar Kruskal-Wallis com n >= 5)
    lotacoes = ["Setor Operacional", "Administrativo", "Gestão e Projetos"]
    perfis_lotacao = {
        "Setor Operacional": {"eot": 2.6, "ind": 3.6, "col": 2.7, "sofr": 3.5, "dano": 3.4},
        "Administrativo": {"eot": 3.5, "ind": 2.8, "col": 3.4, "sofr": 2.4, "dano": 2.3},
        "Gestão e Projetos": {"eot": 4.1, "ind": 2.2, "col": 4.2, "sofr": 1.8, "dano": 1.7},
    }

    sexos = ["Masculino", "Feminino"]
    escolaridades = ["Superior Incompleto", "Superior Completo", "Pós-Graduação"]
    estados_civis = ["Solteiro", "Casado/União Estável", "Divorciado/Separado"]
    contratos = ["Efetivo - Concursado", "Efetivo - CLT", "Estagiário", "Terceirizado"]
    problemas_saude = ["Nenhum", "1 ou 2", "3 ou mais"]

    linhas = []
    for i in range(1, n_respondentes + 1):
        lot = lotacoes[(i - 1) % len(lotacoes)]
        p = perfis_lotacao[lot]

        linha = {
            "respondent_id": f"SIM_{i:02d}",
            "idade": random.randint(20, 58),
            "sexo": random.choice(sexos),
            "escolaridade": random.choice(escolaridades),
            "estado_civil": random.choice(estados_civis),
            "cargo_atual": f"Analista / Técnico {i % 4 + 1}",
            "lotacao": lot,
            "tempo_instituicao_meses": random.randint(3, 180),
            "tempo_cargo_meses": random.randint(2, 120),
            "tipo_contrato": random.choice(contratos),
            "problemas_saude_ultimo_ano": random.choice(problemas_saude),
        }

        for cod in ITENS_ORDENADOS:
            if cod.startswith("eot_"):
                alvo = p["eot"]
            elif cod.startswith("eeg_"):
                num = int(cod.split("_")[1])
                alvo = p["ind"] if num <= 10 else p["col"]
            elif cod.startswith("eist_"):
                alvo = p["sofr"]
            else:  # edt_
                alvo = p["dano"]

            linha[cod] = gerar_likert_centrado(alvo, dispersao=0.85)

        linhas.append(linha)

    df = pd.DataFrame(linhas, columns=COLUNAS_CSV)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"Geradas {len(df)} respostas simuladas em '{CSV_PATH}'.")


if __name__ == "__main__":
    gerar_dados(n_respondentes=30, seed=42)