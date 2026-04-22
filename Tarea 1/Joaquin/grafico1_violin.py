import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Carga y limpieza ────────────────────────────────────────────────
df = pd.read_csv("bustabit.csv")
cashed = df["CashedOut"].dropna()

# Recortar en el percentil 99 para que el violin sea legible
# (valores extremos como 126x distorsionan la escala)
p99 = cashed.quantile(0.99)
cashed_clipped = cashed[cashed <= p99]

# Crear 4 grupos de riesgo para el violin múltiple
bins   = [1.0, 1.5, 2.0, 5.0, p99]
labels = ["Muy conservador\n(1x – 1.5x)",
          "Conservador\n(1.5x – 2x)",
          "Moderado\n(2x – 5x)",
          "Arriesgado\n(5x+)"]

cashed_df = cashed_clipped.to_frame()
cashed_df["grupo"] = pd.cut(cashed_clipped, bins=bins, labels=labels, include_lowest=True)

grupos_data = [cashed_df[cashed_df["grupo"] == lbl]["CashedOut"].values for lbl in labels]
pcts = [len(g) / len(cashed_clipped) * 100 for g in grupos_data]

# ── Paleta y estilo ─────────────────────────────────────────────────
DARK_BG   = "#0d0d0d"
CARD_BG   = "#161616"
ACCENT    = "#f5c518"        # amarillo casino
COLORS    = ["#4cc9f0", "#4361ee", "#f72585", "#ff9f1c"]
TEXT      = "#e8e8e8"
GRID      = "#2a2a2a"

plt.rcParams.update({
    "font.family": "monospace",
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})

fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
ax.set_facecolor(CARD_BG)

# ── Violin plots ────────────────────────────────────────────────────
parts = ax.violinplot(
    grupos_data,
    positions=range(1, 5),
    showmedians=False,
    showextrema=False,
    widths=0.7,
)

for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(COLORS[i])
    pc.set_edgecolor(COLORS[i])
    pc.set_alpha(0.75)

# Box plot interior delgado
bp = ax.boxplot(
    grupos_data,
    positions=range(1, 5),
    widths=0.08,
    patch_artist=True,
    medianprops=dict(color=ACCENT, linewidth=2.5),
    boxprops=dict(facecolor="#ffffff22", edgecolor="#ffffff55", linewidth=1),
    whiskerprops=dict(color="#ffffff55", linewidth=1),
    capprops=dict(color="#ffffff55", linewidth=1),
    flierprops=dict(marker="o", markersize=1.5, alpha=0.2,
                    markerfacecolor="#ffffff44", markeredgecolor="none"),
    showfliers=False,
)

# Puntos de mediana destacados
for i, data in enumerate(grupos_data):
    med = np.median(data)
    ax.scatter(i + 1, med, color=ACCENT, s=60, zorder=5, linewidths=0)

# ── Anotaciones de porcentaje ───────────────────────────────────────
for i, (pct, data) in enumerate(zip(pcts, grupos_data)):
    ax.text(i + 1, max(data) + 0.15, f"{pct:.1f}%\nde jugadores",
            ha="center", va="bottom", fontsize=8.5,
            color=COLORS[i], fontweight="bold")

# ── Línea de referencia en 1x ───────────────────────────────────────
ax.axhline(1.0, color="#ffffff22", linestyle="--", linewidth=1)

# ── Ejes y etiquetas ────────────────────────────────────────────────
ax.set_xticks(range(1, 5))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Multiplicador de salida (CashedOut)", fontsize=10, labelpad=10)
ax.set_ylim(0.85, cashed_clipped.max() + 0.5)
ax.yaxis.grid(True, color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.tick_params(length=0)

# ── Título ──────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         "¿A qué multiplicador retiran los jugadores de Bustabit?",
         ha="center", fontsize=15, fontweight="bold", color=TEXT)
fig.text(0.5, 0.91,
         "Distribución del CashedOut por grupo de riesgo  ·  Datos: bustabit.csv  ·  Percentil 99 = {:.1f}x".format(p99),
         ha="center", fontsize=8.5, color="#888888")

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("grafico1_violin.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print("Guardado: grafico1_violin.png")
