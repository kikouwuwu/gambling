import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Carga y limpieza ────────────────────────────────────────────────
df = pd.read_csv("bustabit.csv")
df_won = df[df["CashedOut"].notna()].copy()

# Escala log para ambos ejes (elimina ceros y outliers extremos)
df_won = df_won[(df_won["Bet"] > 0) & (df_won["CashedOut"] > 1)]

log_bet     = np.log10(df_won["Bet"])
log_cashed  = np.log10(df_won["CashedOut"])

# Recortar percentiles extremos
q_bet_hi    = log_bet.quantile(0.99)
q_cash_hi   = log_cashed.quantile(0.99)
mask = (log_bet <= q_bet_hi) & (log_cashed <= q_cash_hi)
log_bet    = log_bet[mask]
log_cashed = log_cashed[mask]

# ── Paleta y estilo ─────────────────────────────────────────────────
DARK_BG  = "#0d0d0d"
CARD_BG  = "#161616"
ACCENT   = "#f5c518"
TEXT     = "#e8e8e8"
GRID     = "#2a2a2a"

plt.rcParams.update({
    "font.family": "monospace",
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})

fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
ax.set_facecolor(CARD_BG)

# ── Hexbin ──────────────────────────────────────────────────────────
hb = ax.hexbin(
    log_bet, log_cashed,
    gridsize=55,
    cmap="inferno",
    mincnt=1,
    linewidths=0.2,
)

# Barra de color
cb = fig.colorbar(hb, ax=ax, pad=0.02)
cb.set_label("Nº de jugadas", fontsize=9, color=TEXT)
cb.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=TEXT, fontsize=8)

# Línea de tendencia (regresión lineal en espacio log)
z = np.polyfit(log_bet, log_cashed, 1)
p = np.poly1d(z)
x_line = np.linspace(log_bet.min(), log_bet.max(), 200)
ax.plot(x_line, p(x_line), color=ACCENT, linewidth=2, linestyle="--",
        label=f"Tendencia (pendiente={z[0]:.3f})", zorder=5)
ax.legend(fontsize=8.5, framealpha=0.2, labelcolor=TEXT,
          facecolor="#222222", edgecolor="#444444")

# ── Ejes en escala original (potencias de 10) ───────────────────────
x_ticks = [0, 1, 2, 3, 4, 5, 6]
y_ticks = [0, 0.3, 0.7, 1.0, 1.3, 1.7]

ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{10**t:,.0f}" for t in x_ticks], fontsize=8)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{10**t:.1f}x" for t in y_ticks], fontsize=8)

ax.set_xlabel("Apuesta — Bet (escala log₁₀, en bits)", fontsize=10, labelpad=8)
ax.set_ylabel("Multiplicador de salida — CashedOut (escala log₁₀)", fontsize=10, labelpad=8)

ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.tick_params(length=0)

# ── Anotaciones de zonas ────────────────────────────────────────────
ax.text(0.15, 1.5, "Jugadas pequeñas\narriesgadas", fontsize=7.5,
        color="#ffffff66", ha="left", style="italic")
ax.text(4.5, 0.12, "Jugadas grandes\nconservadoras", fontsize=7.5,
        color="#ffffff66", ha="left", style="italic")

# ── Título ──────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         "¿Apostar más implica arriesgar más en Bustabit?",
         ha="center", fontsize=15, fontweight="bold", color=TEXT)
fig.text(0.5, 0.91,
         "Densidad de jugadas ganadas: Apuesta vs Multiplicador de salida  ·  Datos: bustabit.csv  ·  n = {:,}".format(len(log_bet)),
         ha="center", fontsize=8.5, color="#888888")

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("grafico2_hexbin.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print("Guardado: grafico2_hexbin.png")
