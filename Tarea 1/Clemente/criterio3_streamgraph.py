"""
Integrante 2 — Criterio 3: Volumen de apuestas y participación a lo largo del tiempo
Fuente de datos: Bustabit Gambling Game Dataset (Kaggle)
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
Dataset: bustabit.csv — 50,000 registros de apuestas (oct–dic 2016)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

# ─── Carga y preparación de datos ───────────────────────────────────────────
df = pd.read_csv("bustabit.csv")
df["PlayDate"] = pd.to_datetime(df["PlayDate"], utc=True)
df["Date"] = df["PlayDate"].dt.date

# Separar jugadas ganadoras vs perdedoras
df["Won"] = df["Profit"].notna() & (df["Profit"] > 0)

# Agregar por día: total apostado y número de jugadas (ganadas/perdidas)
daily = df.groupby("Date").agg(
    total_bet=("Bet", "sum"),
    plays_won=("Won", "sum"),
    plays_total=("Id", "count"),
).reset_index()

daily["plays_lost"] = daily["plays_total"] - daily["plays_won"]
daily["Date"] = pd.to_datetime(daily["Date"])
daily = daily.sort_values("Date")

# Suavizado con gaussiana para el streamgraph
sigma = 1.2
won_smooth   = gaussian_filter1d(daily["plays_won"].values.astype(float),   sigma)
lost_smooth  = gaussian_filter1d(daily["plays_lost"].values.astype(float),  sigma)
total_smooth = gaussian_filter1d(daily["total_bet"].values.astype(float),   sigma)

# ─── Figura ─────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0d0d1a",
                                gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

x = np.arange(len(daily))
dates = daily["Date"].values

# --- Streamgraph superior: jugadas ganadas / perdidas ---
baseline = -(won_smooth + lost_smooth) / 2
y_won_bot  = baseline
y_won_top  = baseline + won_smooth
y_lost_bot = y_won_top
y_lost_top = y_lost_bot + lost_smooth

ax1.set_facecolor("#0d0d1a")

ax1.fill_between(x, y_won_bot, y_won_top,  color="#00e5a0", alpha=0.85, linewidth=0)
ax1.fill_between(x, y_lost_bot, y_lost_top, color="#ff4f6b", alpha=0.80, linewidth=0)
ax1.fill_between(x, y_won_bot, y_won_top,  color="#00e5a0", alpha=0.25,
                 linewidth=0)  # brillo suave

# Línea central de referencia
ax1.axhline(0, color="white", linewidth=0.4, alpha=0.3, linestyle="--")

# Etiquetas de fechas en eje X
tick_idx = np.linspace(0, len(daily)-1, 8, dtype=int)
ax1.set_xticks(tick_idx)
ax1.set_xticklabels([], fontsize=0)
ax1.set_yticks([])
ax1.set_xlim(0, len(daily)-1)
ax1.spines[:].set_visible(False)

ax1.set_title("Dinámica temporal de Bustabit (oct–dic 2016)",
              fontsize=15, fontweight="bold", color="white", pad=14)
ax1.text(0.01, 0.95, "Jugadas por día",
         transform=ax1.transAxes, fontsize=9, color="white", alpha=0.6, va="top")

legend_handles = [
    mpatches.Patch(color="#00e5a0", label="Jugadas ganadas"),
    mpatches.Patch(color="#ff4f6b", label="Jugadas perdidas"),
]
ax1.legend(handles=legend_handles, loc="upper right",
           framealpha=0, fontsize=9, labelcolor="white")

# --- Área inferior: total apostado en bits ---
ax2.set_facecolor("#0d0d1a")

ax2.fill_between(x, 0, total_smooth / 1e6,
                 color="#7b8ef7", alpha=0.75, linewidth=0)
ax2.plot(x, total_smooth / 1e6,
         color="#a9b8ff", linewidth=1.4, alpha=0.9)

ax2.set_xticks(tick_idx)
ax2.set_xticklabels(
    [pd.Timestamp(dates[i]).strftime("%d %b") for i in tick_idx],
    fontsize=8, color="#aaaacc"
)
ax2.set_xlim(0, len(daily)-1)
ax2.set_ylabel("Millones\nde bits", fontsize=8, color="#aaaacc", rotation=0,
               labelpad=40, va="center")
ax2.yaxis.set_label_coords(-0.045, 0.5)
ax2.tick_params(axis="y", colors="#aaaacc", labelsize=7)
ax2.tick_params(axis="x", colors="#aaaacc")
ax2.spines["bottom"].set_color("#333355")
ax2.spines["left"].set_color("#333355")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_facecolor("#0d0d1a")
ax2.text(0.01, 0.92, "Total apostado (bits)",
         transform=ax2.transAxes, fontsize=9, color="white", alpha=0.6, va="top")

# Fuente
fig.text(0.99, 0.01,
         "Fuente: Bustabit Gambling Dataset · Kaggle (2016)",
         ha="right", fontsize=7, color="#666688", style="italic")


plt.savefig("criterio3_streamgraph.png", dpi=180, bbox_inches="tight", facecolor="#0d0d1a")
print("Guardado: criterio3_streamgraph.png")