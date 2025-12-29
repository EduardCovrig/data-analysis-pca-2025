import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#1. PRELUARE DATE
#skiprows=4 deoarece primele 4 randuri nu contin date
df_gdp = pd.read_csv("GDP_PER_CAPITA.csv", skiprows=4)
df_forest = pd.read_csv("FOREST_AREA.csv", skiprows=4)
df_land = pd.read_csv("LAND_AREA.csv", skiprows=4)

#2. PRELUCRARE (Anul 2023)
gdp = df_gdp[['Country Name', 'Country Code', '2023']].rename(columns={'2023': 'GDP_per_Capita'})
forest = df_forest[['Country Code', '2023']].rename(columns={'2023': 'Forest_Area'})
land = df_land[['Country Code', '2023']].rename(columns={'2023': 'Land_Area'})

# Unim tabelele
df = gdp.merge(forest, on='Country Code').merge(land, on='Country Code')

#3. CALCULE
df['Forest_Percentage'] = (df['Forest_Area'] / df['Land_Area']) * 100
df['Log_Land_Area'] = np.log10(df['Land_Area'])

# Eliminam valorile lipsa si erorile
df_clean = df.dropna()
df_clean = df_clean[(df_clean['Forest_Percentage'] <= 100) & (df_clean['Land_Area'] > 0)]

# Selectam doar coloanele numerice pentru analiza
features = ['GDP_per_Capita', 'Forest_Percentage', 'Log_Land_Area']
x_orig = df_clean[features].values
m = len(features) # numar variabile (3)

#4. STANDARDIZARE DATE
x = (x_orig - np.mean(x_orig, axis=0)) / np.std(x_orig, axis=0)

#5. APLICARE PCA
pca = PCA()
pca.fit(x)
alpha = pca.explained_variance_ # Valori proprii
c = pca.transform(x) # Componentele principale (Scorurile)

#6. CRITERII DE SELECTIE-
# Kaiser (Alpha > 1)
kaiser_crit = np.where(alpha > 1)[0]
print(f"Componente semnificative (Kaiser > 1): {len(kaiser_crit)}")

# Varianta Explicata
print(f"Varianta explicata (%): {pca.explained_variance_ratio_ * 100}")

# 7. MATRICEA DE CORELATII
# Calculam corelatia exacta intre X (date originale) si C (componente)
corr_matrix = np.corrcoef(x, c, rowvar=False)
# Extragem doar randurile variabilelor, coloanele componentelor
loadings = corr_matrix[:m, m:]

# Afisam tabelul pentru primele 2 componente (PC1, PC2)
loadings_df = pd.DataFrame(loadings[:, :2], columns=['PC1', 'PC2'], index=features)
print("\nMatricea Corelatiilor Factoriale (Loadings):")
print(loadings_df)

#8. PARTEA DE GRAFICE

# Grafic 1: Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 4), alpha, 'ro-', linewidth=2)
plt.axhline(1, color='blue', linestyle='--', label='Kaiser (Alpha=1)')
plt.title('Scree Plot - Valorile Proprii')
plt.xlabel('Componenta')
plt.ylabel('Valoare Proprie')
plt.xticks([1, 2, 3])
plt.legend()
plt.grid()
plt.savefig('scree_plot_2023.png') #

# Grafic 2: Corelograma
plt.figure(figsize=(8, 8))
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
# Desenare cerc
circle = plt.Circle((0, 0), 1, color='blue', fill=False)
plt.gca().add_artist(circle)

# Desenare sageti pentru corelatii
for i, var in enumerate(features):
    x_val = loadings[i, 0] # Corelatia cu PC1
    y_val = loadings[i, 1] # Corelatia cu PC2
    plt.arrow(0, 0, x_val, y_val, color='red', head_width=0.05)
    plt.text(x_val * 1.15, y_val * 1.15, var, color='black', fontsize=11)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Corelograma (Cercul Corelatiilor)')
plt.grid()
plt.savefig('biplot_2023.png')

print("\nAnaliza completa. Imaginile au fost salvate.")