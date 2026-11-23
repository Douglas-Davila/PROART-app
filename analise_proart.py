import json
import numpy as np
import pandas as pd
from scipy import stats


def cronbach_alpha(df_itens: pd.DataFrame) -> float:
    """Calcula o Alfa de Cronbach para um subconjunto de itens numéricos."""
    clean = df_itens.dropna(axis=0, how="any")
    k = clean.shape[1]
    if k < 2 or clean.shape[0] < 2:
        return np.nan

    var_itens = clean.var(axis=0, ddof=1).sum()
    var_total = clean.sum(axis=1).var(axis=0, ddof=1)
    if var_total == 0 or np.isnan(var_total):
        return np.nan

    return float((k / (k - 1)) * (1.0 - (var_itens / var_total)))


def classificar_escore(valor: float, faixas: list[dict]) -> str:
    """Classifica um escore médio segundo as faixas configuradas no JSON."""
    if pd.isna(valor):
        return "Sem dados"
    for faixa in faixas:
        if faixa["min"] <= valor <= faixa["max"]:
            return faixa["rotulo"]
    return "Fora da faixa"


def carregar_e_validar(csv_path: str, config_path: str) -> tuple[pd.DataFrame, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(csv_path)
    min_val = config["escala_likert"]["min"]
    max_val = config["escala_likert"]["max"]

    itens_ordenados = []
    for escala_info in config["escalas"].values():
        for fator_info in escala_info["fatores"].values():
            for codigo_item in fator_info["itens"].keys():
                if codigo_item not in itens_ordenados:
                    itens_ordenados.append(codigo_item)

    ausentes = [col for col in itens_ordenados if col not in df.columns]
    if ausentes:
        raise ValueError(f"Colunas de itens ausentes no CSV: {ausentes}")

    for col in itens_ordenados:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        fora = df[col].dropna().apply(lambda x: x < min_val or x > max_val)
        if fora.any():
            raise ValueError(f"Valor fora de [{min_val}, {max_val}] na coluna '{col}'.")

    soma_extremos = min_val + max_val
    for escala_info in config["escalas"].values():
        for fator_info in escala_info["fatores"].values():
            for col in fator_info.get("invertidos", []):
                df[col] = soma_extremos - df[col]

    return df, config


def analisar_proart(df: pd.DataFrame, config: dict) -> dict:
    linhas_fatores = []
    linhas_escalas = []

    for nome_escala, escala_info in config["escalas"].items():
        faixas = escala_info.get("faixas", [])
        itens_escala = []
        cols_fatores_escala = []

        for nome_fator, fator_info in escala_info["fatores"].items():
            itens = list(fator_info["itens"].keys())
            itens_escala.extend(itens)

            col_fator = f"fator__{nome_fator}"
            cols_fatores_escala.append(col_fator)

            df[col_fator] = df[itens].mean(axis=1, skipna=True)
            serie_fator = df[col_fator].dropna()
            media_fator = float(serie_fator.mean()) if len(serie_fator) > 0 else np.nan
            dp_fator = float(serie_fator.std(ddof=1)) if len(serie_fator) > 1 else np.nan

            linhas_fatores.append({
                "escala": nome_escala,
                "fator": nome_fator,
                "n": int(serie_fator.count()),
                "media": media_fator,
                "dp": dp_fator,
                "classificacao": classificar_escore(media_fator, faixas),
                "cronbach_alpha": cronbach_alpha(df[itens]),
                "alpha_ref": fator_info.get("alpha_referencia", np.nan),
            })

        itens_escala = list(dict.fromkeys(itens_escala))
        col_escala = f"escala__{nome_escala}"
        df[col_escala] = df[cols_fatores_escala].mean(axis=1, skipna=True)
        serie_escala = df[col_escala].dropna()
        media_escala = float(serie_escala.mean()) if len(serie_escala) > 0 else np.nan
        dp_escala = float(serie_escala.std(ddof=1)) if len(serie_escala) > 1 else np.nan

        linhas_escalas.append({
            "escala": nome_escala,
            "n": int(serie_escala.count()),
            "media": media_escala,
            "dp": dp_escala,
            "classificacao": classificar_escore(media_escala, faixas),
            "cronbach_alpha": cronbach_alpha(df[itens_escala]),
        })

    return {
        "df_pontuado": df,
        "fatores": pd.DataFrame(linhas_fatores),
        "escalas": pd.DataFrame(linhas_escalas),
    }


def comparar_grupos(df: pd.DataFrame, variavel_numerica: str, variavel_grupo: str, min_n: int = 5) -> dict:
    """
    Aplica Mann-Whitney U (2 grupos) ou Kruskal-Wallis (>= 3 grupos)
    apenas para grupos com tamanho amostral >= min_n.
    """
    sub = df[[variavel_grupo, variavel_numerica]].dropna()
    contagens = sub.groupby(variavel_grupo)[variavel_numerica].count()
    grupos_validos = contagens[contagens >= min_n].index.tolist()

    amostras = [
        sub.loc[sub[variavel_grupo] == g, variavel_numerica].to_numpy()
        for g in grupos_validos
    ]

    if len(amostras) < 2:
        return {
            "variavel": variavel_numerica,
            "grupo": variavel_grupo,
            "teste": None,
            "estatistica": np.nan,
            "p_valor": np.nan,
            "grupos_incluidos": grupos_validos,
        }

    if len(amostras) == 2:
        stat, p_val = stats.mannwhitneyu(amostras[0], amostras[1], alternative="two-sided")
        nome_teste = "Mann-Whitney U"
    else:
        stat, p_val = stats.kruskal(*amostras)
        nome_teste = "Kruskal-Wallis"

    return {
        "variavel": variavel_numerica,
        "grupo": variavel_grupo,
        "teste": nome_teste,
        "estatistica": float(stat),
        "p_valor": float(p_val),
        "grupos_incluidos": grupos_validos,
    }