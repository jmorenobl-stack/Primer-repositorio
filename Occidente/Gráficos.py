#%% Paquetes necesarios
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%%Ruta Relativa
plt.style.use('seaborn-v0_8-whitegrid')
archivo = "Variables.xlsx"
def get_df(hoja, columnas):
    # Extrae solo las columnas necesarias, limpia nulos/textos y convierte tipos de datos en 5 líneas
    df = pd.read_excel(archivo, sheet_name=hoja, skiprows=9).iloc[:, :len(columnas)]
    df.columns = columnas
    df = df[df['Fecha'] != 'Fecha'].dropna(subset=['Fecha'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df[columnas[1:]] = df[columnas[1:]].apply(pd.to_numeric, errors='coerce')
    return df

#%%Carga de datos
df_inf = get_df('3', ['Fecha', 'Inf_Nuc'])
df_tcol = get_df('6', ['Fecha', 'T_Col'])
df_tus = get_df('7', ['Fecha', 'T_US'])

df_pib = get_df('5', ['Fecha', 'PIB']).sort_values('Fecha')
df_pib['YoY'] = df_pib['PIB'].pct_change(4) * 100 # Variación interanual

df_macro = pd.merge(df_inf, df_tcol, on='Fecha')
df_tasas = pd.merge(df_tcol, df_tus, on='Fecha')

#%% Gráfico #1 Efectividad de la política monetaria
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(df_macro['Fecha'], df_macro['Inf_Nuc'], "#1100FF", lw=2.5, label='Inflación')

ax2 = ax1.twinx()
ax2.plot(df_macro['Fecha'], df_macro['T_Col'], '#C8102E', lw=2.5, label='Tasa BanRep')

ax1.set(ylabel='Variación Interanual (%)', title='Respuesta de la Política Monetaria frente a la Inflación en Colombia')
ax2.set(ylabel='Tasa Referencia (%)')
# Fusión elegante de leyendas de ambos ejes en una sola línea
ax1.legend(*[a+b for a,b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())], loc='upper left')
plt.show()

#%% Gráfico #2 Crecimiento PIB Colombia
fig, ax = plt.subplots(figsize=(11, 5))
colores = ['#1100FF' if x >= 0 else '#C8102E' for x in df_pib['YoY']]
ax.bar(df_pib['Fecha'], df_pib['YoY'], width=80, color=colores, alpha=0.85)
ax.axhline(0, color='black', lw=1.2)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set(ylabel='Variación Interanual (%)', title='Crecimiento PIB Colombia')
plt.show()

#%% Gráfico #3 Diferencial de Tasas
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df_tasas['Fecha'], df_tasas['T_Col'], '#C8102E', lw=2.5, label='BanRep (COL)')
ax.plot(df_tasas['Fecha'], df_tasas['T_US'], "#6E6E6E", lw=2.5, ls='-.', label='Fed (US)')
ax.fill_between(df_tasas['Fecha'], df_tasas['T_Col'], df_tasas['T_US'], color='#5DA2D5', alpha=0.15)

ax.set(ylabel='Tasa (%)', title='Diferencial Tasas: Colombia vs. EE. UU.')
ax.legend(loc='upper right')
plt.show()

#%% Gráfico #4: PIB, CONSUMO E INVERSIÓN (Base 100 = Q4 2019)
# Usando la función 'get_df' que definimos previamente
df_pib_col = get_df('5', ['Fecha', 'PIB_Indice', 'Consumo_Indice', 'Inversion_Indice'])
# Las columnas del índice están en las posiciones 7, 8 y 9. Ajustamos la selección
df_pib_col = pd.read_excel(archivo, sheet_name='5', skiprows=10).iloc[:, [0, 7, 8, 9]]
df_pib_col.columns = ['Fecha', 'PIB', 'Consumo', 'Inversion']
df_pib_col = df_pib_col[df_pib_col['Fecha'] != 'Fecha'].dropna(subset=['Fecha'])
df_pib_col['Fecha'] = pd.to_datetime(df_pib_col['Fecha'])

for col in ['PIB', 'Consumo', 'Inversion']:
    df_pib_col[col] = pd.to_numeric(df_pib_col[col], errors='coerce')

# Configuración del gráfico #4
import datetime
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(df_pib_col['Fecha'], df_pib_col['PIB'], color='#1100FF', lw=3, label='PIB')
ax.plot(df_pib_col['Fecha'], df_pib_col['Consumo'], color='#2CA02C', lw=2.5, ls='--', label='Consumo (HyG)')
ax.plot(df_pib_col['Fecha'], df_pib_col['Inversion'], color='#C8102E', lw=2.5, ls='-.', label='Inversión (FBC)')

# Añadir línea de referencia (Base 100) y punto exacto de Dic 2019
ax.axhline(100, color='black', lw=1.2, alpha=0.8)
ax.axvline(datetime.datetime(2019, 12, 31), color='grey', ls=':', lw=1.5, alpha=0.7)
ax.scatter([datetime.datetime(2019, 12, 31)], [100], color='black', s=50, zorder=5) # Punto negro en 100

ax.set(ylabel='Índice (Base 100 = Q4 2019)', title='Evolución del PIB, Consumo e Inversión en Colombia')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.show()
