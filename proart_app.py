import base64
import fcntl
import io
import json
import os
import re
import textwrap
import threading
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import streamlit as st

from analise_proart import analisar_proart, carregar_e_validar, cronbach_alpha

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "proart_config.json")
CSV_PATH = os.path.join(BASE_DIR, "respostas_proart.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CSV_LOCK = threading.Lock()

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

st.set_page_config(
    page_title="PRO Tools — Questionário PROART",
    page_icon="protools.png",
    layout="wide"
)

def img_to_base64(caminho_arquivo: str) -> str:
    with open(caminho_arquivo, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_b64 = img_to_base64("protools.png")

st.markdown(
    f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 10px 18px;
        margin-bottom: 14px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background-color: rgba(255, 255, 255, 0.03);
    ">
        <img src="data:image/png;base64,{logo_b64}" alt="PRO Tool" style="width: 42px; height: 42px; border-radius: 8px; object-fit: contain;">
        <div>
            <div style="font-size: 1.35rem; font-weight: 700; line-height: 1.15;">PRO Tool</div>
            <div style="font-size: 0.85rem; opacity: 0.8;">Aplicação do Protocolo de Avaliação PROART</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

OPCOES_SEXO = [
    "Homem",
    "Mulher",
    "Pessoa não-binária",
    "Outro",
    "Prefiro não informar",
]

OPCOES_ESCOLARIDADE = [
    "Ensino Fundamental",
    "Ensino Médio",
    "Ensino Superior Incompleto",
    "Ensino Superior Completo",
    "Pós-Graduação / Especialização",
    "Mestrado / Doutorado",
]

OPCOES_ESTADO_CIVIL = [
    "Solteiro(a)",
    "Casado(a) / União Estável",
    "Divorciado(a) / Separado(a)",
    "Viúvo(a)",
]

OPCOES_CONTRATO = [
    "Efetivo / Concursado",
    "CLT",
    "Contrato Temporário",
    "Estágio / Bolsa",
    "Terceirizado / Prestação de Serviço",
    "Outro",
]

OPCOES_SAUDE = [
    "Nenhum",
    "1 ou 2",
    "3 ou mais",
]

ESTILOS_ESCALAS = {
    "EOT_Organizacao_do_Trabalho": {
        "titulo": "EOT — Organização do Trabalho",
        "sigla": "EOT",
        "borda": "#2563eb",
        "fundo": "#eff6ff",
        "badge_bg": "#dbeafe",
        "badge_fg": "#1e40af",
        "descricao": "Avalia divisão das tarefas e divisão social do trabalho.",
    },
    "EEG_Estilos_de_Gestao": {
        "titulo": "EEG — Estilos de Gestão",
        "sigla": "EEG",
        "borda": "#7c3aed",
        "fundo": "#f5f3ff",
        "badge_bg": "#ede9fe",
        "badge_fg": "#5b21b6",
        "descricao": "Avalia características dos estilos de gestão individualista e coletivista.",
    },
    "EIST_Indicadores_de_Sofrimento": {
        "titulo": "EIST — Indicadores de Sofrimento no Trabalho",
        "sigla": "EIST",
        "borda": "#d97706",
        "fundo": "#fffbeb",
        "badge_bg": "#fef3c7",
        "badge_fg": "#92400e",
        "descricao": "Avalia falta de sentido do trabalho, esgotamento mental e falta de reconhecimento.",
    },
    "EDT_Danos_Relacionados_ao_Trabalho": {
        "titulo": "EDT — Danos Relacionados ao Trabalho",
        "sigla": "EDT",
        "borda": "#dc2626",
        "fundo": "#fef2f2",
        "badge_bg": "#fee2e2",
        "badge_fg": "#991b1b",
        "descricao": "Avalia danos psicológicos, sociais e físicos associados ao trabalho.",
    },
}


def obter_senha_resultados() -> str:
    try:
        if "RESULTADOS_SENHA" in st.secrets:
            return str(st.secrets["RESULTADOS_SENHA"])
    except Exception:
        pass
    return os.environ.get("PROART_RESULTADOS_SENHA", "proart2026")


def obter_escalas_estruturadas(config: dict) -> list[dict]:
    blocos = []
    for nome_escala, escala_info in config["escalas"].items():
        estilo = ESTILOS_ESCALAS.get(
            nome_escala,
            {
                "titulo": nome_escala.replace("_", " "),
                "sigla": nome_escala.split("_")[0],
                "borda": "#0284c7",
                "fundo": "#f0f9ff",
                "badge_bg": "#e0f2fe",
                "badge_fg": "#075985",
                "descricao": "",
            },
        )
        fatores_lista = []
        total_itens_escala = 0
        for nome_fator, fator_info in escala_info["fatores"].items():
            itens_fator = [
                {"codigo": cod, "texto": txt}
                for cod, txt in fator_info["itens"].items()
            ]
            total_itens_escala += len(itens_fator)
            fatores_lista.append({
                "nome": nome_fator,
                "titulo": nome_fator.replace("_", " "),
                "itens": itens_fator,
            })
        blocos.append({
            "nome": nome_escala,
            "titulo": estilo["titulo"],
            "sigla": estilo["sigla"],
            "borda": estilo["borda"],
            "fundo": estilo["fundo"],
            "badge_bg": estilo["badge_bg"],
            "badge_fg": estilo["badge_fg"],
            "descricao": estilo["descricao"],
            "total_itens": total_itens_escala,
            "fatores": fatores_lista,
        })
    return blocos


BLOCOS_ESCALAS = obter_escalas_estruturadas(CONFIG)
ITENS_PROART = [
    (item["codigo"], item["texto"], bloco["nome"], fator["nome"])
    for bloco in BLOCOS_ESCALAS
    for fator in bloco["fatores"]
    for item in fator["itens"]
]
COLUNAS_CSV = ["respondent_id"] + CAMPOS_DEMO + [cod for cod, _, _, _ in ITENS_PROART]


def localizar_logo(nome_base: str) -> str | None:
    candidatos = [
        f"{nome_base}.png",
        f"{nome_base}.jpg",
        f"{nome_base}.jpeg",
        f"{nome_base}.webp",
        f"logo_{nome_base}.png",
    ]
    if nome_base == "ceng":
        candidatos.extend(["logo_centro_engenharias.png", "logo_engenharia.png"])

    for pasta in [STATIC_DIR, BASE_DIR]:
        for nome in candidatos:
            caminho = os.path.join(pasta, nome)
            if os.path.isfile(caminho):
                return caminho
    return None


def garantir_csv() -> None:
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        pd.DataFrame(columns=COLUNAS_CSV).to_csv(CSV_PATH, index=False, encoding="utf-8")


def salvar_resposta_segura(registro: dict) -> None:
    with CSV_LOCK:
        garantir_csv()
        with open(CSV_PATH, "a+", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            try:
                f.seek(0)
                df_atual = pd.read_csv(f)
                df_novo = pd.concat([df_atual, pd.DataFrame([registro])], ignore_index=True)
                f.seek(0)
                f.truncate()
                df_novo.to_csv(f, index=False, encoding="utf-8")
                f.flush()
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


def normalizar_numero_inteiro(valor) -> int | None:
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        if pd.isna(valor):
            return None
        return int(valor)
    texto = str(valor).strip()
    if not texto:
        return None
    match = re.search(r"-?\d+", texto)
    return int(match.group(0)) if match else None


def normalizar_tempo_meses(valor, unidade: str = "meses") -> int | None:
    num = normalizar_numero_inteiro(valor)
    if num is None or num < 0:
        return None
    return num * 12 if unidade == "anos" else num


def aplicar_estilo_r() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman", "Times"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.1,
        "axes.grid": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    })


def fig_para_png_bytes(fig, dpi: int = 220) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def grafico_problemas_saude_r(df: pd.DataFrame) -> bytes | None:
    col = "problemas_saude_ultimo_ano"
    if col not in df.columns:
        return None
    serie = df[col].dropna().astype(str).str.strip()
    serie = serie[serie != ""]
    if serie.empty:
        return None

    aplicar_estilo_r()
    contagens = serie.value_counts()
    ordem = ["1 ou 2", "Nenhum", "3 ou mais"]
    categorias = [c for c in ordem if c in contagens.index] + [
        c for c in contagens.index if c not in ordem
    ]
    valores = [int(contagens[c]) for c in categorias]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    barras = ax.bar(categorias, valores, width=0.58, color="#595959", edgecolor="none")

    max_val = max(valores) if valores else 1
    ax.set_ylim(0, max_val * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Number of work-related health problems", fontsize=12.5, labelpad=8)
    ax.set_ylabel("Number of participants", fontsize=12.5, labelpad=8)
    ax.tick_params(axis="both", labelsize=11.5, length=4, width=1.1)

    for barra, val in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2.0,
            val + max_val * 0.02,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    return fig_para_png_bytes(fig)


def grafico_distribuicao_lotacao_r(df: pd.DataFrame) -> bytes | None:
    col = "lotacao"
    if col not in df.columns:
        return None
    serie = df[col].dropna().astype(str).str.strip()
    serie = serie[serie != ""]
    if serie.empty:
        return None

    aplicar_estilo_r()
    contagens = serie.value_counts().sort_values(ascending=True)
    categorias = contagens.index.tolist()
    valores = [int(v) for v in contagens.values]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    barras = ax.barh(categorias, valores, height=0.62, color="#595959", edgecolor="none")

    max_val = max(valores) if valores else 1
    ax.set_xlim(0, max_val * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Number of participants", fontsize=12, labelpad=8)
    ax.tick_params(axis="both", labelsize=11, length=4, width=1.1)

    for barra, val in zip(barras, valores):
        ax.text(
            val + max_val * 0.015,
            barra.get_y() + barra.get_height() / 2.0,
            f"{val}",
            ha="left",
            va="center",
            fontsize=12,
        )

    return fig_para_png_bytes(fig)


def cor_hex_celula(valor: float | None, tipo_escala: str) -> tuple[str, str]:
    if valor is None or pd.isna(valor):
        return "#ffffff", "#111827"
    v = float(valor)
    if tipo_escala == "risco_direto":
        if v >= 3.70:
            return "#2e9e5b", "#ffffff"
        elif v >= 2.30:
            return "#f4b400", "#ffffff"
        else:
            return "#e53935", "#ffffff"
    elif tipo_escala == "risco_inverso":
        if v <= 2.29:
            return "#2e9e5b", "#ffffff"
        elif v <= 3.69:
            return "#f4b400", "#ffffff"
        else:
            return "#e53935", "#ffffff"
    else:
        if v >= 3.51:
            return "#e53935", "#ffffff"
        elif v >= 2.50:
            return "#f4b400", "#ffffff"
        else:
            return "#2e9e5b", "#ffffff"


def _classificar_valor(media_val: float | None, tipo_escala: str) -> str:
    if media_val is None or pd.isna(media_val):
        return "-"
    v = float(media_val)
    if tipo_escala == "risco_direto":
        if v >= 3.70:
            return "Risco Baixo"
        elif v >= 2.30:
            return "Médio Risco"
        else:
            return "Risco Grave"
    elif tipo_escala == "risco_inverso":
        if v <= 2.29:
            return "Risco Baixo"
        elif v <= 3.69:
            return "Médio Risco"
        else:
            return "Risco Grave"
    else:
        if v >= 3.51:
            return "Característico"
        elif v >= 2.50:
            return "Moderado"
        else:
            return "Pouco Característico"


def renderizar_tabela_png(
    colunas: list[str],
    linhas: list[list[dict]],
    larguras_relativas: list[float],
    super_header: tuple[str, int, int] | None = None,
    destaque_ultimas_n: int = 0,
    largura_total_pol: float = 12.0,
    fontsize_base: float = 8.5,
) -> bytes:
    aplicar_estilo_r()
    n_rows = len(linhas)

    total_w = sum(larguras_relativas)
    norm_w = [w / total_w for w in larguras_relativas]

    linhas_processadas = []
    alturas_linhas = []

    alturas_linhas.append(0.42)
    if super_header:
        alturas_linhas.append(0.38)

    for idx_r, linha in enumerate(linhas):
        eh_destaque = destaque_ultimas_n > 0 and idx_r >= (n_rows - destaque_ultimas_n)
        max_quebras = 1
        celulas_proc = []

        for idx_c, cel in enumerate(linha):
            txt_orig = str(cel.get("texto", ""))
            larg_pol = norm_w[idx_c] * largura_total_pol
            chars_por_linha = max(5, int((larg_pol - 0.14) / (fontsize_base * 0.0072)))
            align = cel.get("align", "left")

            if align == "left" and len(txt_orig) > chars_por_linha:
                txt_wrapped = textwrap.fill(txt_orig, width=chars_por_linha, break_long_words=True)
            else:
                txt_wrapped = txt_orig

            n_lin = txt_wrapped.count("\n") + 1
            if n_lin > max_quebras:
                max_quebras = n_lin

            cel_copia = dict(cel)
            cel_copia["texto_wrap"] = txt_wrapped
            celulas_proc.append(cel_copia)

        altura_linha = max(0.36, max_quebras * 0.22 + 0.12)
        if eh_destaque and idx_r == n_rows - destaque_ultimas_n:
            altura_linha += 0.04
        alturas_linhas.append(altura_linha)
        linhas_processadas.append(celulas_proc)

    altura_total_pol = sum(alturas_linhas) + 0.15
    fig, ax = plt.subplots(figsize=(largura_total_pol, altura_total_pol))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, altura_total_pol)
    ax.axis("off")

    x_starts = []
    acc = 0.0
    for w in norm_w:
        x_starts.append((acc, w))
        acc += w

    def linha_horizontal(x_ini: float, x_fim: float, y_pos: float, cor: str = "#d1d5db", lw: float = 0.8):
        ax.plot([x_ini, x_fim], [y_pos, y_pos], color=cor, linewidth=lw, solid_capstyle="butt", clip_on=True)

    def desenhar_celula(
        x0,
        w,
        y_top,
        h,
        texto,
        bg=None,
        fg="#111827",
        bold=False,
        align="left",
        fs=fontsize_base,
    ):
        if bg and bg.lower() != "#ffffff":
            rect = Rectangle(
                (x0, y_top - h),
                w,
                h,
                facecolor=bg,
                edgecolor="none",
                linewidth=0.0,
                clip_on=True,
            )
            ax.add_patch(rect)

        if align == "center":
            tx = x0 + w / 2.0
            ha = "center"
        elif align == "right":
            tx = x0 + w - 0.008
            ha = "right"
        else:
            tx = x0 + 0.008
            ha = "left"

        ax.text(
            tx,
            y_top - h / 2.0,
            str(texto),
            ha=ha,
            va="center",
            color=fg,
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            clip_on=True,
        )

    y_atual = altura_total_pol - 0.07
    idx_alt = 0

    linha_horizontal(0.0, 1.0, y_atual, cor="#6b7280", lw=1.1)

    if super_header:
        h_sup = alturas_linhas[idx_alt]
        idx_alt += 1
        titulo_sup, idx_ini, span = super_header
        x_sup = sum(norm_w[:idx_ini])
        w_sup = sum(norm_w[idx_ini : idx_ini + span])
        desenhar_celula(
            x_sup,
            w_sup,
            y_atual,
            h_sup,
            titulo_sup,
            bg=None,
            fg="#111827",
            bold=True,
            align="center",
            fs=fontsize_base + 0.5,
        )
        y_atual -= h_sup
        linha_horizontal(x_sup, x_sup + w_sup, y_atual, cor="#9ca3af", lw=0.8)

    h_head = alturas_linhas[idx_alt]
    idx_alt += 1
    for idx_c, col_nome in enumerate(colunas):
        x0, w = x_starts[idx_c]
        alinh = "left" if idx_c < 2 and not super_header else ("left" if idx_c == 0 else "center")
        desenhar_celula(
            x0,
            w,
            y_atual,
            h_head,
            col_nome,
            bg=None,
            fg="#111827",
            bold=True,
            align=alinh,
            fs=fontsize_base,
        )
    y_atual -= h_head
    linha_horizontal(0.0, 1.0, y_atual, cor="#6b7280", lw=1.1)

    for idx_r, linha in enumerate(linhas_processadas):
        h_row = alturas_linhas[idx_alt]
        idx_alt += 1
        eh_destaque = destaque_ultimas_n > 0 and idx_r >= (n_rows - destaque_ultimas_n)
        eh_ultima = idx_r == (n_rows - 1)

        if eh_destaque and idx_r == (n_rows - destaque_ultimas_n):
            linha_horizontal(0.0, 1.0, y_atual, cor="#374151", lw=1.1)

        for idx_c, cel in enumerate(linha):
            x0, w = x_starts[idx_c]
            txt = cel.get("texto_wrap", cel.get("texto", ""))
            bg = cel.get("bg")
            fg = cel.get("fg", "#111827")
            bold = cel.get("bold", False) or eh_destaque
            align = cel.get("align", "left")
            desenhar_celula(
                x0,
                w,
                y_atual,
                h_row,
                txt,
                bg=bg,
                fg=fg,
                bold=bold,
                align=align,
                fs=fontsize_base,
            )
        y_atual -= h_row
        cor_div = "#6b7280" if eh_ultima else "#e5e7eb"
        lw_div = 1.1 if eh_ultima else 0.75
        linha_horizontal(0.0, 1.0, y_atual, cor=cor_div, lw=lw_div)

    return fig_para_png_bytes(fig, dpi=220)


def montar_tabela_caracteristicas(df_raw: pd.DataFrame) -> list[dict]:
    total = len(df_raw)
    linhas = [{"Characteristic": "Participants", "Value": f"{total}"}]

    for campo, rotulo in [
        ("idade", "Age (years)"),
        ("tempo_instituicao_meses", "Tenure in institution (months)"),
        ("tempo_cargo_meses", "Tenure in current position (months)"),
    ]:
        if campo in df_raw.columns:
            s = pd.to_numeric(df_raw[campo], errors="coerce").dropna()
            if len(s) > 0:
                dp = s.std(ddof=1) if len(s) > 1 else 0.0
                linhas.append({"Characteristic": rotulo, "Value": f"{s.mean():.2f} ± {dp:.2f}"})

    for campo, prefixo in [
        ("sexo", "Gender"),
        ("lotacao", "Department"),
        ("problemas_saude_ultimo_ano", "Work-related health problems"),
    ]:
        if campo in df_raw.columns:
            s = df_raw[campo].dropna().astype(str).str.strip()
            s = s[s != ""]
            vc = s.value_counts()
            for cat, cnt in vc.items():
                pct = (cnt / total * 100.0) if total > 0 else 0.0
                linhas.append({"Characteristic": f"{prefixo}: {cat}", "Value": f"{int(cnt)} ({pct:.1f}%)"})

    return linhas


def montar_tabelas_proart(df_pontuado: pd.DataFrame, config: dict):
    linhas_alpha = []
    linhas_consolidado = []
    tabelas_fatores = []
    max_itens_fator = 0

    for nome_escala, escala_info in config["escalas"].items():
        tipo_esc = escala_info.get("tipo_classificacao", "risco_inverso")
        sigla_esc = nome_escala.split("_")[0]

        for nome_fator, fator_info in escala_info["fatores"].items():
            itens_dict = fator_info["itens"]
            cods_itens = list(itens_dict.keys())
            max_itens_fator = max(max_itens_fator, len(cods_itens))

            col_fator = f"fator__{nome_fator}"
            serie_fator = df_pontuado[col_fator].dropna() if col_fator in df_pontuado.columns else pd.Series(dtype=float)
            media_fator = float(serie_fator.mean()) if len(serie_fator) > 0 else np.nan
            dp_fator = float(serie_fator.std(ddof=1)) if len(serie_fator) > 1 else np.nan
            cv_fator = float((dp_fator / media_fator) * 100.0) if pd.notna(media_fator) and media_fator != 0 and pd.notna(dp_fator) else np.nan

            alpha_val = cronbach_alpha(df_pontuado[cods_itens])
            linhas_alpha.append({
                "Scale": f"{nome_fator.replace('_', ' ')} ({sigla_esc})",
                "Alpha": f"{alpha_val:.3f}" if pd.notna(alpha_val) else "-",
            })

            medias_itens = []
            linhas_questoes = []
            for idx, cod in enumerate(cods_itens, start=1):
                s_item = pd.to_numeric(df_pontuado[cod], errors="coerce").dropna()
                m_item = float(s_item.mean()) if len(s_item) > 0 else np.nan
                medias_itens.append(m_item)
                linhas_questoes.append({
                    "Questão": f"{idx}. {itens_dict[cod]}",
                    "Média": f"{m_item:.2f}" if pd.notna(m_item) else "-",
                    "Classificação": _classificar_valor(m_item, tipo_esc),
                })

            linhas_consolidado.append({
                "Escala": sigla_esc,
                "Fator": nome_fator.replace("_", " "),
                "Media_num": media_fator,
                "Media_fmt": f"{media_fator:.2f}" if pd.notna(media_fator) else "-",
                "DP_fmt": f"{dp_fator:.2f}" if pd.notna(dp_fator) else "-",
                "Tipo_Escala": tipo_esc,
                "Itens_medias": medias_itens,
            })

            img_fator = renderizar_tabela_png(
                colunas=["Questão", "Média", "Classificação do Risco"],
                linhas=[
                    [
                        {"texto": q["Questão"], "align": "left"},
                        {"texto": q["Média"], "align": "center"},
                        {"texto": q["Classificação"], "align": "center"},
                    ]
                    for q in linhas_questoes
                ]
                + [
                    [
                        {"texto": "Média", "align": "left", "bold": True},
                        {"texto": f"{media_fator:.2f}" if pd.notna(media_fator) else "-", "align": "center", "bold": True},
                        {"texto": "", "align": "center"},
                    ],
                    [
                        {"texto": "Desvio Padrão", "align": "left"},
                        {"texto": f"{dp_fator:.2f}" if pd.notna(dp_fator) else "-", "align": "center"},
                        {"texto": "", "align": "center"},
                    ],
                    [
                        {"texto": "Coeficiente de Variação (%)", "align": "left"},
                        {"texto": f"{cv_fator:.2f}" if pd.notna(cv_fator) else "-", "align": "center"},
                        {"texto": "", "align": "center"},
                    ],
                ],
                larguras_relativas=[8.5, 1.4, 2.4],
                destaque_ultimas_n=3,
                largura_total_pol=12.0,
                fontsize_base=8.8,
            )

            tabelas_fatores.append({
                "arquivo_png": f"Tabela_{sigla_esc}_{nome_fator}.png",
                "img_bytes": img_fator,
                "titulo": f"{sigla_esc} — {nome_fator.replace('_', ' ')}",
            })

    img_carac = renderizar_tabela_png(
        colunas=["Characteristic", "Value"],
        linhas=[
            [
                {"texto": r["Characteristic"], "align": "left"},
                {"texto": r["Value"], "align": "center"},
            ]
            for r in montar_tabela_caracteristicas(df_pontuado)
        ],
        larguras_relativas=[7.0, 2.5],
        largura_total_pol=9.5,
        fontsize_base=9.0,
    )

    img_alpha = renderizar_tabela_png(
        colunas=["Scale", "α de Cronbach"],
        linhas=[
            [
                {"texto": r["Scale"], "align": "left"},
                {"texto": r["Alpha"], "align": "center"},
            ]
            for r in linhas_alpha
        ],
        larguras_relativas=[7.5, 2.2],
        largura_total_pol=9.5,
        fontsize_base=9.0,
    )

    cols_cons = ["Escala", "Fator de Escala", "Média", "Desvio Padrão"] + [f"Q{i}" for i in range(1, max_itens_fator + 1)]
    larg_cons = [1.1, 4.2, 1.1, 1.3] + [1.05] * max_itens_fator
    linhas_cons_png = []
    for r in linhas_consolidado:
        bg_m, fg_m = cor_hex_celula(r["Media_num"], r["Tipo_Escala"])
        row_cells = [
            {"texto": r["Escala"], "align": "left"},
            {"texto": r["Fator"], "align": "left"},
            {"texto": r["Media_fmt"], "align": "center", "bg": bg_m, "fg": fg_m, "bold": True},
            {"texto": r["DP_fmt"], "align": "center"},
        ]
        for i in range(max_itens_fator):
            if i < len(r["Itens_medias"]) and pd.notna(r["Itens_medias"][i]):
                val_q = r["Itens_medias"][i]
                bg_q, fg_q = cor_hex_celula(val_q, r["Tipo_Escala"])
                row_cells.append({"texto": f"{val_q:.2f}", "align": "center", "bg": bg_q, "fg": fg_q, "bold": True})
            else:
                row_cells.append({"texto": "", "align": "center"})
        linhas_cons_png.append(row_cells)

    img_consolidado = renderizar_tabela_png(
        colunas=cols_cons,
        linhas=linhas_cons_png,
        larguras_relativas=larg_cons,
        super_header=("Média de Cada Questão", 4, max_itens_fator),
        largura_total_pol=20.0,
        fontsize_base=8.0,
    )

    return {
        "caracteristicas": img_carac,
        "alpha": img_alpha,
        "consolidado": img_consolidado,
    }, tabelas_fatores


def aplicar_estilos_frontend() -> None:
    st.markdown(
        """
        <style>
            .proart-faixa-logos {
                display: flex;
                align-items: center;
                gap: 28px;
                margin-bottom: 16px;
                padding: 12px 16px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                flex-wrap: wrap;
            }
            .proart-faixa-logos img {
                max-height: 96px;
                width: auto;
                object-fit: contain;
            }
            .proart-hero {
                background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
                border: 1px solid #dbeafe;
                border-radius: 14px;
                padding: 20px 24px;
                margin-bottom: 16px;
            }
            .proart-hero h1 {
                margin: 0 0 6px 0;
                font-size: 1.85rem;
                color: #0f172a;
            }
            .proart-autores {
                color: #334155;
                font-size: 0.98rem;
                margin-bottom: 4px;
            }
            .proart-subtitulo {
                color: #64748b;
                font-size: 0.88rem;
                margin: 0;
            }
            .proart-secao-demo {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 6px solid #0f172a;
                border-radius: 12px;
                padding: 16px 20px;
                margin-top: 12px;
                margin-bottom: 14px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            .proart-escala-box {
                border-radius: 12px;
                padding: 16px 20px;
                margin-top: 26px;
                margin-bottom: 14px;
                box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
            }
            .proart-escala-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                flex-wrap: wrap;
                margin-bottom: 6px;
            }
            .proart-escala-titulo {
                margin: 0;
                font-size: 1.25rem;
                font-weight: 700;
                color: #0f172a;
            }
            .proart-escala-badge {
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 700;
            }
            .proart-escala-desc {
                margin: 0;
                font-size: 0.94rem;
                color: #334155;
            }
            .proart-fator-titulo {
                margin-top: 16px;
                margin-bottom: 8px;
                padding-bottom: 4px;
                border-bottom: 1px solid #e2e8f0;
                color: #1e293b;
                font-weight: 700;
                font-size: 1.04rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_faixa_logos() -> None:
    logo_ufpel = localizar_logo("ufpel")
    logo_labserg = localizar_logo("labserg")
    logo_ceng = localizar_logo("ceng")

    def img_tag(caminho: str | None, alt: str) -> str:
        if not caminho:
            return ""
        ext = os.path.splitext(caminho)[1].lower().lstrip(".")
        mime = "jpeg" if ext == "jpg" else ext
        with open(caminho, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f'<img src="data:image/{mime};base64,{b64}" alt="{alt}">'

    tags = [t for t in [
        img_tag(logo_ufpel, "UFPel"),
        img_tag(logo_labserg, "LabSErg"),
        img_tag(logo_ceng, "CEng"),
    ] if t]

    if tags:
        st.markdown(
            '<div class="proart-faixa-logos">' + "".join(tags) + "</div>",
            unsafe_allow_html=True,
        )


garantir_csv()
aplicar_estilos_frontend()
renderizar_faixa_logos()

aba_form, aba_res = st.tabs(["Questionário PROART", "Resultados (Acesso Restrito)"])

with aba_form:
    st.markdown(
        """
        <div class="proart-hero">
            <h1>Protocolo de Avaliação PROART</h1>
            <div class="proart-autores"><strong>Autores:</strong> Renata Heidtmann Bemnvenuti e Douglas Fernandes D'avila</div>
            <p class="proart-subtitulo">Instrumento estruturado em 91 itens distribuídos em 4 escalas e 10 fatores (Facas, 2021).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Como preencher e utilizar", expanded=True):
        st.markdown(
            "1. **Dados sociodemográficos e laborais:** informe idade e tempos em números, escolhendo *Meses* ou *Anos* (tudo é padronizado em meses).\n"
            "2. **Escala de respostas (1 a 5):** marque de **1 (Nunca)** a **5 (Sempre)** em cada item.\n"
            "3. **Acumulação:** cada envio adiciona uma nova linha e atualiza os cálculos nos resultados."
        )

    respondidos_atuais = sum(
        1 for cod, _, _, _ in ITENS_PROART if st.session_state.get(f"item_{cod}") is not None
    )
    total_itens = len(ITENS_PROART)
    st.progress(respondidos_atuais / total_itens if total_itens else 0.0)
    st.caption(f"Progresso das questões: {respondidos_atuais} de {total_itens} respondidas.")

    with st.form("form_proart", clear_on_submit=True):
        st.markdown(
            """
            <div class="proart-secao-demo">
                <h3 style="margin:0 0 4px 0; font-size:1.15rem; color:#0f172a;">1. Dados Sociodemográficos e Laborais</h3>
                <p style="margin:0; font-size:0.9rem; color:#475569;">Preencha as informações do perfil profissional e do contexto de trabalho.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)

        with c1:
            idade_val = st.number_input("Idade (anos)", min_value=14, max_value=100, step=1, value=None, placeholder="Ex.: 28")
            escolaridade_val = st.selectbox("Escolaridade", [""] + OPCOES_ESCOLARIDADE)
            t_inst_col1, t_inst_col2 = st.columns([2, 1])
            with t_inst_col1:
                tempo_inst_val = st.number_input("Tempo na instituição", min_value=0, max_value=600, step=1, value=None, placeholder="Ex.: 24")
            with t_inst_col2:
                tempo_inst_unid = st.selectbox("Unidade", ["meses", "anos"], key="unid_inst")

        with c2:
            sexo_val = st.radio("Sexo / Identidade", [""] + OPCOES_SEXO, horizontal=False)
            estado_civil_val = st.selectbox("Estado civil", [""] + OPCOES_ESTADO_CIVIL)
            t_cargo_col1, t_cargo_col2 = st.columns([2, 1])
            with t_cargo_col1:
                tempo_cargo_val = st.number_input("Tempo no cargo atual", min_value=0, max_value=600, step=1, value=None, placeholder="Ex.: 12")
            with t_cargo_col2:
                tempo_cargo_unid = st.selectbox("Unidade", ["meses", "anos"], key="unid_cargo")

        with c3:
            cargo_val = st.text_input("Cargo / Função atual", placeholder="Ex.: Analista Administrativo")
            lotacao_val = st.text_input("Lotação / Setor / Departamento", placeholder="Ex.: Administrativo")
            contrato_val = st.selectbox("Tipo de vínculo / contrato", [""] + OPCOES_CONTRATO)
            saude_val = st.radio("Problemas de saúde relacionados ao trabalho (último ano)", [""] + OPCOES_SAUDE)

        respostas_itens = {}
        for escala in BLOCOS_ESCALAS:
            st.markdown(
                f"""
                <div class="proart-escala-box" style="border-left:6px solid {escala['borda']}; background:{escala['fundo']}; border-top:1px solid {escala['borda']}33; border-right:1px solid {escala['borda']}33; border-bottom:1px solid {escala['borda']}33;">
                    <div class="proart-escala-header">
                        <h3 class="proart-escala-titulo">{escala['titulo']}</h3>
                        <span class="proart-escala-badge" style="background:{escala['badge_bg']}; color:{escala['badge_fg']};">{escala['total_itens']} itens</span>
                    </div>
                    <p class="proart-escala-desc">{escala['descricao']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for fator in escala["fatores"]:
                st.markdown(
                    f'<div class="proart-fator-titulo">{fator["titulo"]}</div>',
                    unsafe_allow_html=True,
                )
                for idx_item, item in enumerate(fator["itens"], start=1):
                    respostas_itens[item["codigo"]] = st.radio(
                        f"**{idx_item}.** [{item['codigo']}] {item['texto']}",
                        options=[1, 2, 3, 4, 5],
                        format_func=lambda v: {
                            1: "1 - Nunca",
                            2: "2 - Raramente",
                            3: "3 - Às vezes",
                            4: "4 - Frequentemente",
                            5: "5 - Sempre",
                        }[v],
                        horizontal=True,
                        index=None,
                        key=f"item_{item['codigo']}",
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        enviado = st.form_submit_button("Salvar Resposta", type="primary", use_container_width=True)

    if enviado:
        faltantes = [cod for cod, v in respostas_itens.items() if v is None]
        if faltantes:
            st.error(f"Existem {len(faltantes)} item(ns) sem resposta. Preencha todos os 91 itens antes de salvar.")
        else:
            idade_norm = normalizar_numero_inteiro(idade_val)
            t_inst_norm = normalizar_tempo_meses(tempo_inst_val, tempo_inst_unid)
            t_cargo_norm = normalizar_tempo_meses(tempo_cargo_val, tempo_cargo_unid)

            registro = {
                "respondent_id": str(uuid.uuid4())[:8],
                "idade": idade_norm if idade_norm is not None else np.nan,
                "sexo": sexo_val.strip(),
                "escolaridade": escolaridade_val.strip(),
                "estado_civil": estado_civil_val.strip(),
                "cargo_atual": cargo_val.strip(),
                "lotacao": lotacao_val.strip(),
                "tempo_instituicao_meses": t_inst_norm if t_inst_norm is not None else np.nan,
                "tempo_cargo_meses": t_cargo_norm if t_cargo_norm is not None else np.nan,
                "tipo_contrato": contrato_val.strip(),
                "problemas_saude_ultimo_ano": saude_val.strip(),
            }
            registro.update(respostas_itens)
            salvar_resposta_segura(registro)
            st.success("Resposta salva com sucesso!")

with aba_res:
    if "autenticado_resultados" not in st.session_state:
        st.session_state["autenticado_resultados"] = False

    if not st.session_state["autenticado_resultados"]:
        st.subheader("Acesso Restrito aos Resultados")
        senha_digitada = st.text_input("Digite a senha de acesso", type="password")
        if st.button("Entrar"):
            if senha_digitada == obter_senha_resultados():
                st.session_state["autenticado_resultados"] = True
                st.rerun()
            else:
                st.error("Senha incorreta.")
    else:
        col_top1, col_top2 = st.columns([4, 1])
        with col_top1:
            st.subheader("Resultados Consolidados PROART")
        with col_top2:
            if st.button("Bloquear / Sair"):
                st.session_state["autenticado_resultados"] = False
                st.rerun()

        df_raw = pd.read_csv(CSV_PATH)
        if df_raw.empty:
            st.warning("Ainda não há respostas registradas.")
        else:
            with open(CSV_PATH, "rb") as f_csv:
                st.download_button(
                    label="Baixar CSV Bruto Acumulado",
                    data=f_csv.read(),
                    file_name="respostas_proart.csv",
                    mime="text/csv",
                )

            df_validado, cfg = carregar_e_validar(CSV_PATH, CONFIG_PATH)
            res = analisar_proart(df_validado, cfg)
            df_pontuado = res["df_pontuado"]

            g_col1, g_col2 = st.columns(2)
            img_saude = grafico_problemas_saude_r(df_pontuado)
            img_lotacao = grafico_distribuicao_lotacao_r(df_pontuado)

            with g_col1:
                if img_saude:
                    st.image(img_saude, width=520)
                    st.download_button(
                        "Exportar Gráfico de Saúde (PNG)",
                        data=img_saude,
                        file_name="Figure_2_Work_related_health_problems.png",
                        mime="image/png",
                    )

            with g_col2:
                if img_lotacao:
                    st.image(img_lotacao, width=520)
                    st.download_button(
                        "Exportar Gráfico de Lotação (PNG)",
                        data=img_lotacao,
                        file_name="Figure_1_Department_distribution.png",
                        mime="image/png",
                    )

            imgs_tab, tabs_fatores = montar_tabelas_proart(df_pontuado, cfg)

            st.markdown("### Tabelas Formatadas (Exportação em PNG)")
            st.image(imgs_tab["caracteristicas"], width=680)
            st.download_button(
                "Exportar Características da Amostra (PNG)",
                data=imgs_tab["caracteristicas"],
                file_name="Table_1_Characteristics.png",
                mime="image/png",
            )

            st.image(imgs_tab["alpha"], width=680)
            st.download_button(
                "Exportar Alfa de Cronbach (PNG)",
                data=imgs_tab["alpha"],
                file_name="Table_2_Cronbach_Alpha.png",
                mime="image/png",
            )

            st.image(imgs_tab["consolidado"], width=960)
            st.download_button(
                "Exportar Consolidado PROART (PNG)",
                data=imgs_tab["consolidado"],
                file_name="Tabela_PROART_Consolidado.png",
                mime="image/png",
            )

            for tab in tabs_fatores:
                st.markdown(f"#### {tab['titulo']}")
                st.image(tab["img_bytes"], width=780)
                st.download_button(
                    f"Exportar {tab['titulo']} (PNG)",
                    data=tab["img_bytes"],
                    file_name=tab["arquivo_png"],
                    mime="image/png",
                    key=f"dl_{tab['arquivo_png']}",
                )
