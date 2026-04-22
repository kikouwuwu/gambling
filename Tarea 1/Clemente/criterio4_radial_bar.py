"""
Integrante 2 — Criterio 4: Patrón de actividad por hora del día
Fuente de datos: Bustabit Gambling Game Dataset (Kaggle)
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
Dataset: bustabit.csv — 50,000 registros de apuestas (oct–dic 2016)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os

# ─── Carga y preparación ────────────────────────────────────────────────────
df = pd.read_csv("bustabit.csv")
df["PlayDate"] = pd.to_datetime(df["PlayDate"], utc=True)
df["Hour"] = df["PlayDate"].dt.hour

hourly = df.groupby("Hour").agg(
    plays=("Id", "count"),
    total_bet=("Bet", "sum"),
    avg_bet=("Bet", "mean"),
).reset_index()

# ─── Coordenadas polares ─────────────────────────────────────────────────────
N = 24
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
width = (2 * np.pi) / N * 0.82       # ancho de cada barra (con pequeño gap)

plays_vals   = hourly["plays"].values.astype(float)
totbet_vals  = hourly["total_bet"].values.astype(float)
avgbet_vals  = hourly["avg_bet"].values.astype(float)

# Normalizar apuestas promedio para el color
norm = Normalize(vmin=avgbet_vals.min(), vmax=avgbet_vals.max())
colors = cm.plasma(norm(avgbet_vals))

# ─── Figura ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 11), facecolor="#0d0d1a")
ax = fig.add_subplot(111, projection="polar")
ax.set_facecolor("#0d0d1a")

# Radio externo: número de jugadas (normalizado)
max_plays = plays_vals.max()
radii = plays_vals / max_plays          # 0–1

bars = ax.bar(
    theta, radii, width=width,
    bottom=0.12,                        # hueco central
    color=colors, alpha=0.88,
    edgecolor="#0d0d1a", linewidth=0.6,
    zorder=3
)

# Anillo de referencia en 50 % y 100 %
for r_ref, label in [(0.5, "50%"), (1.0, "100%")]:
    circle_theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(circle_theta, np.full_like(circle_theta, r_ref + 0.12),
            color="white", linewidth=0.4, alpha=0.18, zorder=2)

# ─── Etiquetas de horas ──────────────────────────────────────────────────────
hour_labels = [
    "0h", "1h", "2h", "3h", "4h", "5h",
    "6h", "7h", "8h", "9h", "10h", "11h",
    "12h", "13h", "14h", "15h", "16h", "17h",
    "18h", "19h", "20h", "21h", "22h", "23h",
]
ax.set_xticks(theta)
ax.set_xticklabels(hour_labels, fontsize=8.5, color="#ccccee")
ax.tick_params(axis="x", pad=6)

# Marcadores cardinales destacados
for h, label in [(0, "Medianoche"), (6, "6 AM"), (12, "Mediodía"), (18, "6 PM")]:
    angle = theta[h]
    ax.annotate(
        label,
        xy=(angle, 1.12 + 0.12),
        xytext=(angle, 1.30 + 0.12),
        ha="center", va="center",
        fontsize=8, color="#ffe0a0", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#ffe0a080", lw=0.8)
    )

# Ocultar ejes de radio
ax.set_yticks([])
ax.spines["polar"].set_visible(False)

# ─── Hora pico anotada ───────────────────────────────────────────────────────
peak_hour = int(plays_vals.argmax())
ax.annotate(
    f"Peak: {peak_hour}h\n({int(plays_vals[peak_hour])} jugadas)",
    xy=(theta[peak_hour], radii[peak_hour] + 0.12),
    xytext=(theta[peak_hour], radii[peak_hour] + 0.38),
    ha="center", fontsize=8.5, color="#00e5a0", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#00e5a0", lw=1.2)
)

# ─── Colorbar (apuesta promedio) ─────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                    fraction=0.025, pad=0.12, aspect=40,
                    shrink=0.6)
cbar.ax.tick_params(labelcolor="white", labelsize=8)
cbar.set_label("Apuesta promedio (bits)", color="white", fontsize=9)
cbar.outline.set_edgecolor("#555577")
for spine in cbar.ax.spines.values():
    spine.set_edgecolor("#555577")
    
# ─── Textos ─────────────────────────────────────────────────────────────────
ax.set_title("Patrón de actividad horaria — Bustabit (2016)",
             fontsize=14, fontweight="bold", color="white", pad=24, y=1.06)

fig.text(0.5, 0.02,
         "Altura = número de jugadas  ·  Color = apuesta promedio\n"
         "Fuente: Bustabit Gambling Dataset · Kaggle (2016)",
         ha="center", fontsize=8, color="#888899", style="italic")

plt.subplots_adjust(bottom=0.15)
plt.savefig("criterio4_radial_bar.png", dpi=180, bbox_inches="tight", facecolor="#0d0d1a")
print("Guardado: criterio4_radial_bar.png")