import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Carga de datos ---
try:
    df_transacciones = pd.read_csv('ventas_tienda_limpio.csv')
    print("Datos de transacciones cargados correctamente.")
    # Convertir 'sale_date' a formato datetime para cálculos de frecuencia
    df_transacciones['sale_date'] = pd.to_datetime(df_transacciones['sale_date'])
except FileNotFoundError:
    print("Error: 'ventas_tienda_limpio.csv' no encontrado. Asegúrate de que esté en el mismo directorio.")
    exit()

# --- 2. Agregación de datos por cliente para obtener las features ---
# Calcula las features de compra por cada customer_id
df_clientes = df_transacciones.groupby('customer_id').agg(
    Monto_Total_Gasto=('total_amour_customer_id', 'sum'), # Suma el monto total de las transacciones por cliente
    Cantidad_Productos=('quantity', 'sum'),               # Suma la cantidad de productos comprados por cliente
    Frecuencia_Compra=('sale_date', 'nunique')            # Número de días únicos que el cliente realizó compras
).reset_index()

# Verificación de las features calculadas
print("\nDataFrame de clientes con features para clustering:")
print(df_clientes.head())

# --- 3. Selección de variables para el clustering ---
features_for_clustering = ['Monto_Total_Gasto', 'Cantidad_Productos', 'Frecuencia_Compra']
X = df_clientes[features_for_clustering]

# --- 4. Estandarización de los Datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nDatos estandarizados.")

# --- 5. Determinación del número óptimo de clústeres (Método del Codo) ---
wcss = [] # Suma de cuadrados intra-clúster (Inercia)
# Prueba K desde 1 hasta un límite razonable, por ejemplo 10
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('WCSS (Inercia)')
plt.grid(True)
plt.savefig('elbow_method_plot.png')
# plt.show() # Descomentar para ver el gráfico al ejecutar
print("Gráfico del Método del Codo guardado como 'elbow_method_plot.png'")

# --- 6. Aplicación del algoritmo K-Means con K óptimo ---
optimal_k = 4 # Basado en tu informe y análisis previo
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans_final.fit(X_scaled)
labels = kmeans_final.labels_
df_clientes['Cluster'] = labels # Añadir etiquetas al DataFrame de clientes

print(f"\nClustering finalizado con K={optimal_k}. Etiquetas asignadas a los clientes.")

# --- 7. Evaluación del modelo con K óptimo ---
# Asegurarse de que X_scaled tenga al menos 2 muestras y K sea menor que el número de muestras
if len(X_scaled) >= optimal_k and optimal_k > 1:
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies = davies_bouldin_score(X_scaled, labels)

    print(f"\nMétricas de Evaluación para K={optimal_k}:")
    print(f"  Silhouette Score: {silhouette:.2f}")
    print(f"  Calinski-Harabasz Index: {calinski:.2f}")
    print(f"  Davies-Bouldin Index: {davies:.2f}")
else:
    print(f"\nNo se pueden calcular métricas para K={optimal_k} con {len(X_scaled)} muestras (K debe ser > 1 y < número de muestras).")


# --- 8. Reducción de Dimensionalidad con PCA y Visualización ---
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=['Componente Principal 1', 'Componente Principal 2'])
pca_df['Cluster'] = labels

varianza_explicada = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Componente Principal 1',
    y='Componente Principal 2',
    hue='Cluster',
    data=pca_df,
    palette='viridis', # o 'plasma', 'cividis', etc.
    legend='full',
    alpha=0.7
)
plt.xlabel(f"Componente Principal 1 ({varianza_explicada[0]:.2f}%)")
plt.ylabel(f"Componente Principal 2 ({varianza_explicada[1]:.2f}%)")
plt.title(f'Visualización de Clústeres (K={optimal_k}) mediante PCA')
plt.grid(True)
plt.savefig('pca_clusters_plot.png')
# plt.show() # Descomentar para ver el gráfico al ejecutar
print(f"Gráfico de PCA con clústeres guardado como 'pca_clusters_plot.png'")

# --- 9. Análisis de Perfiles de Clientes (Opcional: mostrar centroides) ---
# Cálculo de los centroides en la escala original para facilitar la interpretación
centroids_scaled = kmeans_final.cluster_centers_
centroids_original_scale = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original_scale, columns=features_for_clustering)
centroids_df['Cluster'] = range(optimal_k)
print("\nCentroides de los Clústeres (en escala original de las variables):")
print(centroids_df)

# Guardar los centroides en un CSV (opcional)
# centroids_df.to_csv('customer_cluster_centroids.csv', index=False)