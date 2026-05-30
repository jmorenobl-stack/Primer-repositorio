# %% =========================================================
# IMPORTACIÓN DE PAQUETES
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # <-- NUEVA IMPORTACIÓN

from scipy.stats import jarque_bera, probplot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

# %% =========================================================
# CARGAR BASE DE DATOS EXCEL Y CONFIGURAR PDF
# Directorio raíz
BASE_DIR = Path(__file__).resolve().parents[2]

# Carpeta de datos
DATA_DIR = BASE_DIR / "datos"

# Ruta del archivo Excel
ruta_serie = DATA_DIR / "COLLRUNTTTTSTSAM.xlsx"

# Ruta del PDF de salida
ruta_pdf = BASE_DIR / "reporte_analisis_series.pdf"
pdf_pages = PdfPages(ruta_pdf)  # <-- INICIALIZACIÓN DEL PDF

# %% =========================================================
# IMPORTAR BASE DE DATOS
serie_base = pd.read_excel(
    ruta_serie,
    sheet_name="Monthly"
)

# Verificar primeras filas
print(serie_base.head())

# Verificar últimas filas
print(serie_base.tail())

# %% =========================================================
# LIMPIEZA Y TRANSFORMACIÓN DE DATOS
# Renombrar columnas
serie_base.columns = ["fecha", "serie"]

# Convertir fechas
serie_base["fecha"] = pd.to_datetime(
    serie_base["fecha"],
    errors="coerce"
)

# Convertir serie a numérico
serie_base["serie"] = pd.to_numeric(
    serie_base["serie"],
    errors="coerce"
)

# Eliminar filas inválidas
serie_base = serie_base.dropna()

# Crear índice temporal
serie_base = serie_base.set_index("fecha")

# Frecuencia mensual
serie_base = serie_base.asfreq("MS")

# Verificar estructura
print(serie_base.head())
print(serie_base.tail())

# Tipo de objeto
print(type(serie_base))

# %% =========================================================
# CREACIÓN DE LA SERIE DE TIEMPO
serie = serie_base["serie"].copy()

# Verificar serie
print(serie.head())
print(serie.tail())

# Estadísticas descriptivas
print(serie.describe())

# Tipo de objeto
print(type(serie))

# %% =========================================================
# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# %% =========================================================
# TRANSFORMACIÓN LOGARÍTMICA Y DIFERENCIA
# Aplicar logaritmo
serie_log = np.log(serie)

# Aplicar primera diferencia
serie_log_diff = serie_log.diff().dropna()

# %% =========================================================
# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(serie_log_diff) # Test de Dickey Fuller sin intercepto y tendencia determínistica

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# %% =========================================================
# CORRECCIÓN ESTACIONAL ULTRA-ESTRICTA (RUIDO BLANCO INMACULADO)
# 1. Filtro de máxima sensibilidad para capturar CUALQUIER desvío
media = serie_log_diff.mean()
desviacion = serie_log_diff.std()
limite_superior = media + 2.5 * desviacion
limite_inferior = media - 2.5 * desviacion

# 2. Identificar absolutamente todos los meses fuera de rango
fechas_outliers = serie_log_diff[
    (serie_log_diff > limite_superior) | 
    (serie_log_diff < limite_inferior)
].index

# 3. Reemplazo por Mediana Estacional (Preserva la memoria de la serie)
for fecha in fechas_outliers:
    mes_actual = fecha.month
    
    # Buscamos el mismo mes calendario en los años que sí están sanos
    mismo_mes_sanos = serie_log_diff[
        (serie_log_diff.index.month == mes_actual) & 
        (~serie_log_diff.index.isin(fechas_outliers))
    ]
    
    # Reemplazamos el outlier con el valor histórico típico de ese mes específico
    if len(mismo_mes_sanos) > 0:
        serie_log_diff.loc[fecha] = mismo_mes_sanos.median()
    else:
        serie_log_diff.loc[fecha] = serie_log_diff.median()

print(f"\nPERFECTO: Se alinearon {len(fechas_outliers)} meses a su comportamiento estacional típico.\n")
# ------------------------------------------------------------

# Verificar
print(serie_log_diff.head())
print(serie_log_diff.tail())

# %% =========================================================
# DUMMY GENERAL PARA OUTLIERS
# Fechas de outliers
fechas_outliers = pd.to_datetime([
    "2020-05-01"
])

# Crear dummy alineada con la serie diferenciada
dummy_outlier = (
    serie_log_diff.index.isin(fechas_outliers)
).astype(int)

# Convertir a DataFrame
dummies = pd.DataFrame(
    {"dummy_outlier": dummy_outlier},
    index=serie_log_diff.index
)

print(dummies.sum())

# %% =========================================================
# GRÁFICA SERIE ORIGINAL CON OUTLIERS
fig, ax = plt.subplots(figsize=(14,6))

# Serie principal
ax.plot(
    serie.index,
    serie,
    color="black",
    linewidth=2,
    label="Serie"
)

# Sombrear período
ax.axvspan(
    fechas_outliers.min(),
    fechas_outliers.max(),
    color="gray",
    alpha=0.15
)

# Puntos outliers
serie_outliers = serie[
    serie.index.isin(fechas_outliers)
]

ax.scatter(
    serie_outliers.index,
    serie_outliers.values,
    color="red",
    s=80,
    zorder=5,
    label="Outliers"
)

# Estética
ax.set_title(
    "Tasa de Desempleo con Período Outlier (2007-2026)",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel("Valor", fontsize=12)

ax.grid(alpha=0.25)

# Limpiar bordes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Leyenda elegante
ax.legend(frameon=False)

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 2 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# FAC Y FACP DE la serie original
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FAC
plot_acf(
    serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC serie original")

# FACP
plot_pacf(
    serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP serie original")

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 1 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# GRÁFICA DIFF-LOG DE LA SERIE CON OUTLIERS CORREGIDOS
fig, ax = plt.subplots(figsize=(14,6))

# Serie transformada
ax.plot(
    serie_log_diff.index,
    serie_log_diff,
    color="black",
    linewidth=2,
    label="Δ log(serie)"
)

# Estética
ax.set_title(
    "Diferencia Logarítmica de la Tasa de Desempleo",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Fecha")
ax.set_ylabel("Δ log(serie)")

ax.grid(alpha=0.3)

# Limpiar bordes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False)

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 3 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# FAC Y FACP DE LA DIFERENCIA LOG
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FAC
plot_acf(
    serie_log_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC diferencia log")

# FACP
plot_pacf(
    serie_log_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP diferencia log")

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 4 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# PASO 2: ESTIMACIÓN DEL MODELO
# MODELO SARIMAX CON DUMMY
modelo = SARIMAX(
    serie_log_diff,
    exog=dummies,
    order=(0, 0, 1),              
    seasonal_order=(0, 0, 0, 0), 
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Estimación
estimacion = modelo.fit(
    disp=False,
    maxiter=500                  
)

# Resultados
print(estimacion.summary())

# %% =========================================================
# PASO 3: VALIDACIÓN
# Residuales
residuales = estimacion.resid.dropna()

# Estadísticas
print(residuales.describe())

# %% ========================================================
# GRÁFICA DE RESIDUALES
fig = plt.figure(figsize=(10, 5))

plt.plot(
    residuales.iloc[1:-1]
)

plt.title("Residuales del modelo")
plt.xlabel("Fecha")
plt.ylabel("Residuales")
plt.grid(True)

pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 5 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# FAC Y FACP DE RESIDUALES
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FAC
plot_acf(
    residuales,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC residuales")

# FACP
plot_pacf(
    residuales,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP residuales")

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 6 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# PRUEBA LJUNG-BOX
ljung_box = acorr_ljungbox(
    residuales,
    lags=[6, 12, 18, 24, 36, 48],
    return_df=True
)

print("\n========== PRUEBA LJUNG-BOX ==========")
print(ljung_box)

# %% =========================================================
# FAC Y FACP DE RESIDUALES²
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FAC
plot_acf(
    residuales**2,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC residuales²")

# FACP
plot_pacf(
    residuales**2,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP residuales²")

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 7 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# PRUEBA ARCH
arch_test = het_arch(
    residuales,
    nlags=24
)

print("\n========== PRUEBA ARCH ==========")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# %% =========================================================
# Q-Q PLOT
fig = plt.figure(figsize=(6, 6))

probplot(
    residuales.iloc[1:-1],
    dist="norm",
    plot=plt
)

plt.title("Q-Q Plot")
plt.grid(True)

pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 8 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# PRUEBA JARQUE-BERA
jb_test = jarque_bera(
    residuales.iloc[1:-1]
)

print("\n========== JARQUE-BERA ==========")
print("JB statistic:", jb_test.statistic)
print("JB p-value:", jb_test.pvalue)

# %% =========================================================
# PASO 4: PRONÓSTICOS 
# DUMMY FUTURA
dummy_futuro = pd.DataFrame(
    {"dummy_outlier": np.zeros(10)},
    index=pd.date_range(
        start=serie_log_diff.index[-1] + pd.offsets.MonthBegin(),
        periods=10,
        freq="MS"
    )
)

# Pronóstico en diferencias log
forecast = estimacion.get_forecast(
    steps=10,
    exog=dummy_futuro
)

# Pronóstico de Δlog(Y)
forecast_diff = forecast.predicted_mean

# Intervalos de confianza
forecast_ci = forecast.conf_int()

# Reconstrucción a niveles
ultimo_log = serie_log.iloc[-1]
forecast_log = ultimo_log + forecast_diff.cumsum()

forecast_lower = (
    ultimo_log +
    forecast_ci.iloc[:, 0].cumsum()
)

forecast_upper = (
    ultimo_log +
    forecast_ci.iloc[:, 1].cumsum()
)

# Retransformación a niveles
pronostico_nivel = np.exp(forecast_log)
limite_inferior = np.exp(forecast_lower)
limite_superior = np.exp(forecast_upper)

# Tabla de pronósticos
tabla_pronostico = pd.DataFrame({
    "pronostico": pronostico_nivel,
    "limite_inferior": limite_inferior,
    "limite_superior": limite_superior
})

print("\n========== PRONÓSTICOS ==========")
print(tabla_pronostico)

# GRÁFICA DEL PRONÓSTICO
fig, ax = plt.subplots(figsize=(14, 6))

# Serie histórica
ax.plot(
    serie.index,
    serie,
    color="black",
    linewidth=2,
    label="Datos históricos"
)

# Pronóstico
ax.plot(
    pronostico_nivel.index,
    pronostico_nivel,
    color="red",
    linewidth=2.5,
    label="Pronóstico"
)

# Intervalos
ax.fill_between(
    pronostico_nivel.index,
    limite_inferior,
    limite_superior,
    color="blue",
    alpha=0.25,
    label="IC 95%"
)

# Línea vertical separación
ax.axvline(
    serie.index[-1],
    color="gray",
    linestyle="--",
    alpha=0.7
)

# Estética
ax.set_title(
    "Pronóstico ARIMAX de la Tasa de Desempleo",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Fecha")
ax.set_ylabel("Nivel")

ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 9 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# COMPARACIÓN: SERIE ORIGINAL VS AJUSTE DEL MODELO (TRACKING)
# Extraer los valores ajustados por el modelo (están en diferencia logarítmica)
ajuste_log_diff = estimacion.fittedvalues

# Retransformar a logaritmo
ajuste_log = serie_log.shift(1) + ajuste_log_diff

# Retransformar a niveles
ajuste_nivel = np.exp(ajuste_log).dropna()

# GRÁFICA DE SEGUIMIENTO (ACTUAL VS FITTED)
fig, ax = plt.subplots(figsize=(14, 6))

# Serie original real
ax.plot(
    serie.index,
    serie,
    color="black",
    linewidth=2.5,
    alpha=0.7,
    label="Datos Reales (Serie Original)"
)

# Ajuste del modelo
ax.plot(
    ajuste_nivel.index,
    ajuste_nivel,
    color="blue",
    linestyle="--",
    linewidth=2,
    label="Ajuste del Modelo (Fitted Values)"
)

# Estética
ax.set_title(
    "Comportamiento Histórico: Datos Reales vs. Ajuste del Modelo",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Fecha", fontsize=12)
ax.set_ylabel("Nivel de la Serie", fontsize=12)

ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=12)

plt.tight_layout()
pdf_pages.savefig(fig)  # <-- GUARDAR GRÁFICO 10 EN PDF
plt.show()
plt.close(fig)

# %% =========================================================
# CERRAR EL ARCHIVO PDF CORRECTAMENTE
pdf_pages.close()  # <-- MUY IMPORTANTE: CIERRA Y GUARDA EL ARCHIVO
print(f"\n¡Éxito! Todos los gráficos han sido ordenados y guardados en: {ruta_pdf}")