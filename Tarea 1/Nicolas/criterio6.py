import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

# ─── Carga y preparación de datos ───────────────────────────────────────────
df = pd.read_csv("../bustabit.csv") # Asegúrate de que la ruta sea correcta
df["PlayDate"] = pd.to_datetime(df["PlayDate"], utc=True)
df["Hour"] = df["PlayDate"].dt.hour

# Asegurar formato numérico para evitar errores
df["Profit"] = pd.to_numeric(df["Profit"], errors='coerce')

# Filtrar solo jugadas con Profit registrado
df_profit = df[df["Profit"].notna()].copy()

# Limitar profit para mejor visualización (percentil 95) para no deformar el gráfico
profit_95 = df_profit["Profit"].quantile(0.95)
df_profit = df_profit[df_profit["Profit"] <= profit_95].copy()

# ─── Configuración visual (Estilo Premium del Topográfico) ──────────────────
DARK_BG = "#1c1c28"    # Fondo gris azulado elegante y suave
TEXT_COLOR = "#d4d4dc" # Texto en gris claro, menos agresivo a la vista

fig, ax = plt.subplots(figsize=(14, 14), facecolor=DARK_BG,
                       subplot_kw=dict(projection='polar'))
ax.set_facecolor(DARK_BG)

# ─── Conversión a coordenadas polares ──────────────────────────────────────
n_hours = 24
theta_per_hour = 2 * np.pi / n_hours

# Asignar ángulo a cada hora con pequeña dispersión (semilla para estabilidad)
np.random.seed(42)
df_profit["theta_base"] = (df_profit["Hour"] + np.random.random(len(df_profit)) * 0.8 - 0.4) * theta_per_hour

# Radio: normalizar profit
profit_min, profit_max = df_profit["Profit"].min(), df_profit["Profit"].max()

if profit_max == profit_min:
    profit_norm = 0.5
else:
    profit_norm = (df_profit["Profit"] - profit_min) / (profit_max - profit_min)

r_min = 0.1
r_max = 0.9
df_profit["r"] = r_min + profit_norm * (r_max - r_min)

# ─── Colormap Suavizado ───────────────────────────────────────────────────
# Tonos pastel/apagados: Rojo suave (pérdida) -> Gris azulado (neutral) -> Verde suave (ganancia)
colors_palette = ["#e06c75", "#abb2bf", "#98c379"] 
cmap = LinearSegmentedColormap.from_list("profit_elegant", colors_palette)
norm = Normalize(vmin=profit_min, vmax=profit_max)

# ─── Scatter plot circular ────────────────────────────────────────────────
scatter = ax.scatter(df_profit["theta_base"], df_profit["r"],
                     c=df_profit["Profit"], cmap=cmap, norm=norm,
                     s=15, alpha=0.7, edgecolors="none")

# ─── Círculos de referencia y Líneas radiales ─────────────────────────────
radii_ref = np.linspace(r_min, r_max, 5)
for r_ref in radii_ref:
    circle_theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(circle_theta, np.full_like(circle_theta, r_ref),
            color="#ffffff", linewidth=0.5, alpha=0.08, linestyle="--")

for hour in range(24):
    theta = hour * theta_per_hour
    ax.plot([theta, theta], [r_min, r_max], color="#ffffff", linewidth=0.5,
            alpha=0.05, linestyle="--", zorder=1)

# ─── Etiquetas de horas ──────────────────────────────────────────────────
for hour in range(24):
    theta = hour * theta_per_hour
    hour_data = df_profit[df_profit["Hour"] == hour]
    
    if len(hour_data) > 0:
        avg_profit = hour_data["Profit"].mean()
        win_rate = (hour_data["Profit"] > 0).sum() / len(hour_data) * 100
    else:
        avg_profit = 0
        win_rate = 0

    # Etiqueta con hora (Diseño refinado)
    label_r = r_max + 0.12
    ax.text(theta, label_r, f"{hour:02d}:00",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=TEXT_COLOR, bbox=dict(boxstyle="round,pad=0.4",
                                        facecolor='#282a36', edgecolor='#44475a',
                                        alpha=0.8, linewidth=1))

    # Estadísticas en sector
    stats_r = r_min - 0.06
    ax.text(theta, stats_r, f"${avg_profit:.0f}\n{win_rate:.0f}%",
            ha="center", va="center", fontsize=8, color="#888899", alpha=0.8)

# ─── Configurar ejes ──────────────────────────────────────────────────────
ax.set_ylim(0, r_max + 0.25)
ax.set_xticks([]) # Ocultar ticks por defecto
ax.set_yticks([])
ax.spines["polar"].set_color(DARK_BG) # Ocultar borde exterior

# ─── Colorbar ─────────────────────────────────────────────────────────────
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, orientation="horizontal",
                    fraction=0.046, aspect=40, shrink=0.6)
cbar.set_label("Intensidad de Ganancia (Profit en bits)", color=TEXT_COLOR, fontsize=11, labelpad=10)
cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)
cbar.outline.set_edgecolor("#333344")

# ─── Títulos y anotaciones (Estilo Topográfico) ───────────────────────────
plt.suptitle('Beeswarm Circular: Distribución de Ganancias por Hora', 
             y=0.96, color=TEXT_COLOR, fontsize=16, fontweight='bold')
plt.title('Cada punto representa una jugada en su franja horaria respectiva', 
          color='#888899', fontsize=11, style='italic', pad=20)

# Fuente
fig.text(0.02, 0.02,
         "Fuente: Bustabit Gambling Dataset (Elaboración Propia)",
         ha="left", fontsize=9, color="#666677")

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show() # Para guardar: plt.savefig("criterio2_beeswarm_premium.png", dpi=150, facecolor=DARK_BG)