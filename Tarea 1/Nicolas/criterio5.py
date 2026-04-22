import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. Cargar y preparar datos ─────────────────────────────────────────────
df = pd.read_csv('../bustabit.csv')
df['Bet'] = pd.to_numeric(df['Bet'], errors='coerce')
df['CashedOut'] = pd.to_numeric(df['CashedOut'], errors='coerce')

# Limpiar valores nulos
df_clean = df.dropna(subset=['Bet', 'CashedOut'])

# Filtrar para enfocarnos en la zona donde ocurre el 95% de la acción
# (Apuestas hasta 50, Retiros hasta 5x) para evitar que los outliers aplasten el gráfico
df_zoom = df_clean[(df_clean['Bet'] <= 50) & (df_clean['CashedOut'] <= 5)]

# ─── 2. Configuración estética (Colores suaves y elegantes) ─────────────────
DARK_BG = "#1c1c28" # Gris oscuro azulado (relajante)
TEXT_COLOR = "#d4d4dc"

# Crear figura
fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)

# ─── 3. Crear el Gráfico Topográfico (KDE 2D) ──────────────────────────────
# Usamos 'mako' que es una paleta de tonos oscuros y suaves de agua/verde
sns.kdeplot(
    x=df_zoom['Bet'], 
    y=df_zoom['CashedOut'], 
    cmap="mako",         
    fill=True, 
    thresh=0.05,         # Elimina la "basura" visual de las zonas sin datos
    levels=15,           # Cantidad de "terrazas" o curvas de nivel
    ax=ax,
    alpha=0.85         # Ligeramente transparente
)

# ─── 4. Detalles de formato y ejes ─────────────────────────────────────────
# Añadir una grilla sutil
ax.grid(color='#ffffff', alpha=0.05, linestyle='--', linewidth=0.5)

# Personalizar ejes
ax.set_xlabel("Monto Apostado (Bet en bits)", color=TEXT_COLOR, fontsize=12, fontweight='bold')
ax.set_ylabel("Multiplicador de Retiro (CashedOut)", color=TEXT_COLOR, fontsize=12, fontweight='bold')
ax.tick_params(colors=TEXT_COLOR, labelsize=10)

# Ocultar los bordes fuertes
for spine in ax.spines.values():
    spine.set_edgecolor('#333344')

# Títulos y anotaciones
plt.suptitle('Mapa Topográfico del Comportamiento del Jugador', 
             y=0.95, color=TEXT_COLOR, fontsize=16, fontweight='bold')
plt.title('Zonas de mayor concentración (densidad) entre Apuesta y Retiro', 
          color='#888899', fontsize=11, style='italic', pad=15)

# Fuente de datos
plt.text(0.99, 0.01, "Fuente: Bustabit Gambling Dataset (Elaboración Propia)", 
         transform=ax.transAxes, ha='right', color='#666677', fontsize=9)

# Flecha indicativa hacia el pico principal
ax.annotate('Pico de mayor\ncomportamiento\n(Apuestas bajas,\nretiros rápidos)', 
            xy=(5, 1.3), xytext=(15, 2.5),
            arrowprops=dict(facecolor='#a8e6cf', arrowstyle="->", connectionstyle="arc3,rad=.2", color='#a8e6cf'),
            color='#a8e6cf', fontsize=10, fontweight='bold')

# ─── 5. Exportar ───────────────────────────────────────────────────────────
plt.tight_layout()
plt.show() # O plt.savefig('topografico.png', dpi=150, facecolor=DARK_BG)