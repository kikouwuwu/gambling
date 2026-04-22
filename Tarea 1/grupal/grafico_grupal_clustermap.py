"""
CÓDIGO GENERADO POR IA GENERATIVA
Modificaciones realizadas: ver sección 4.3 del informe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ── Carga y agregación por jugador ─────────────────────────────────
df = pd.read_csv("bustabit.csv")
df["won"] = df["CashedOut"].notna().astype(int)

agg = df.groupby("Username").agg(
    n_jugadas      = ("Bet",       "count"),
    apuesta_media  = ("Bet",       "mean"),
    win_rate       = ("won",       "mean"),
    cashout_medio  = ("CashedOut", "mean"),   # NaN ignorados
    profit_total   = ("Profit",    "sum"),    # NaN ignorados
).reset_index()

# Filtrar jugadores con al menos 10 jugadas para mayor robustez
agg = agg[agg["n_jugadas"] >= 10].copy()

# Normalizar profit_total por número de jugadas (profit por jugada)
agg["profit_por_jugada"] = agg["profit_total"] / agg["n_jugadas"]

# Seleccionar métricas para el clustermap
metricas = ["apuesta_media", "win_rate", "cashout_medio",
            "profit_por_jugada", "n_jugadas"]
labels_metricas = [
    "Apuesta\nPromedio",
    "Win Rate",
    "CashedOut\nPromedio",
    "Profit por\nJugada",
    "N° de\nJugadas",
]

# ── Normalización (Z-score) ─────────────────────────────────────────
X = agg[metricas].copy()
# Recortar outliers extremos antes de normalizar (percentil 1-99)
for col in metricas:
    lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = X[col].clip(lo, hi)

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=labels_metricas,
    index=agg["Username"].values,
)

# Limitar a top 60 jugadores por número de jugadas para legibilidad
top_users = agg.nlargest(60, "n_jugadas")["Username"].values
X_top = X_scaled.loc[top_users]

# ── Estilo ──────────────────────────────────────────────────────────
DARK_BG = "#0d0d0d"
CARD_BG = "#161616"
TEXT    = "#e8e8e8"

plt.rcParams.update({
    "font.family":    "monospace",
    "text.color":     TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color":    TEXT,
    "ytick.color":    TEXT,
})

# ── Clustermap ──────────────────────────────────────────────────────
cg = sns.clustermap(
    X_top,
    method       = "ward",
    metric       = "euclidean",
    cmap         = "RdYlGn",
    figsize      = (12, 18),
    linewidths   = 0.3,
    linecolor    = "#2a2a2a",
    dendrogram_ratio = (0.15, 0.05),
    cbar_pos     = (0.02, 0.82, 0.03, 0.15),
    yticklabels  = True,
    xticklabels  = True,
    annot        = False,
    robust       = True,
)

# ── Fondo oscuro en todas las partes ────────────────────────────────
cg.fig.patch.set_facecolor(DARK_BG)
for ax in cg.fig.axes:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

# Colorbar label
cg.cax.set_ylabel("Z-score\nnormalizado", fontsize=8, color=TEXT, labelpad=6)
cg.cax.yaxis.label.set_color(TEXT)
plt.setp(cg.cax.yaxis.get_ticklabels(), color=TEXT, fontsize=7)

# Etiquetas del eje X (métricas)
cg.ax_heatmap.set_xticklabels(
    cg.ax_heatmap.get_xticklabels(),
    rotation=0, fontsize=9, color=TEXT, fontweight="bold"
)
cg.ax_heatmap.set_yticklabels(
    cg.ax_heatmap.get_yticklabels(),
    rotation=0, fontsize=6.5, color="#cccccc"
)

# ── Título ──────────────────────────────────────────────────────────
cg.fig.suptitle(
    "Perfiles de comportamiento de jugadores en Bustabit",
    y=0.995, fontsize=14, fontweight="bold", color=TEXT, fontfamily="monospace"
)
cg.fig.text(
    0.5, 0.978,
    "Clustermap jerárquico (método Ward) sobre métricas normalizadas  ·  Top 60 jugadores más activos  ·  Fuente: bustabit.csv",
    ha="center", fontsize=8, color="#888888", fontfamily="monospace"
)

plt.savefig(
    "grafico_grupal_clustermap.png",
    dpi=150, bbox_inches="tight", facecolor=DARK_BG
)
print("Guardado: grafico_grupal_clustermap.png")
