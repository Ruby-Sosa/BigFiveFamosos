import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Big Five Personality Clusterer", layout="wide")

st.title("🧠 Clasificador de Personalidad (Big Five OCEAN)")
st.write("Ingresa tus puntajes del test Big Five para descubrir tu clúster y compararte con famosos correspondientes.")

# 1. Cargar el dataset, modelo y scaler
@st.cache_resource
def load_assets():
    df = pd.read_csv("https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv")
    kmeans_model = joblib.load("kmeans_BigFive_model.joblib")
    scaler_model = joblib.load("scaler_bigFive.joblib")
    return df, kmeans_model, scaler_model

try:
    df, kmeans, scaler = load_assets()
except Exception as e:
    st.error(f"Error al cargar los recursos: {e}. Asegúrate de tener los archivos 'kmeans_BigFive_model.joblib' y 'scaler_bigFive.joblib' en el mismo directorio.")
    st.stop()

# Obtener etiquetas de categoría si están disponibles o mapearlas
category_mapping = {
    1: 'Actor/actriz',
    2: 'Cantante',
    3: 'Modelo',
    4: 'Tv, series',
    5: 'Radio',
    6: 'Tecnología',
    7: 'Deportes',
    8: 'Politica',
    9: 'Escritor'
}
df['categoria_nombre'] = df['categoria'].map(category_mapping)

# Asegurar que las etiquetas del modelo se asignen al df
if 'cluster' not in df.columns:
    # Aplicar la normalización a todo el dataset para reconstruir las etiquetas si no estuvieran
    features = ['op', 'co', 'ex', 'ag', 'ne']
    df_scaled = scaler.transform(df[features])
    df['cluster'] = kmeans.predict(df_scaled)

# Formulario de entrada
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Introduce tus puntajes (de 0 a 100)")
    op_val = st.slider("Apertura a la experiencia (op)", 0.0, 100.0, 54.9)
    co_val = st.slider("Responsabilidad / Conciencia (co)", 0.0, 100.0, 78.3)
    ex_val = st.slider("Extroversión (ex)", 0.0, 100.0, 35.0)
    ag_val = st.slider("Amabilidad (ag)", 0.0, 100.0, 65.0)
    ne_val = st.slider("Neuroticismo (ne)", 0.0, 100.0, 70.0)

    predict_btn = st.button("Asignar mi Clúster", type="primary")

if predict_btn:
    # 2. Escalar y predecir el clúster para los nuevos datos
    input_data = pd.DataFrame([{ 'op': op_val, 'co': co_val, 'ex': ex_val, 'ag': ag_val, 'ne': ne_val }])
    input_scaled = scaler.transform(input_data)
    assigned_cluster = kmeans.predict(input_scaled)[0]

    with col2:
        st.success(f"🎯 ¡Has sido clasificado en el **Clúster {assigned_cluster}**!")

        # 3. Mostrar el top 5 de famosos de ese clúster
        st.subheader(f"👥 Famosos representativos en el Clúster {assigned_cluster}")
        famous_in_cluster = df[df['cluster'] == assigned_cluster]

        # Mostrar los primeros 5 famosos ordenados de forma predeterminada o por palabra usada
        top_5_famous = famous_in_cluster[['usuario', 'categoria_nombre', 'op', 'co', 'ex', 'ag', 'ne']].head(5)
        st.dataframe(top_5_famous, use_container_width=True)

        # 4. Gráfica de radar
        st.subheader("📊 Comparación en Gráfica de Radar")

        categories = ['op', 'co', 'ex', 'ag', 'ne']
        N = len(categories)

        # Obtener los valores promedio para el clúster asignado
        cluster_avg_values = famous_in_cluster[categories].mean().values.tolist()
        cluster_avg_values += cluster_avg_values[:1]

        # Obtener los valores ingresados por el usuario
        user_values = [op_val, co_val, ex_val, ag_val, ne_val]
        user_values += user_values[:1]

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        plt.xticks(angles[:-1], categories, color='grey', size=11)
        ax.set_rlabel_position(0)

        max_val = max(max(user_values), max(cluster_avg_values))
        limit = int(max_val + 10)

        plt.yticks(list(range(20, limit, 20)), [str(i) for i in range(20, limit, 20)], color="grey", size=8)
        plt.ylim(0, limit)

        # Graficar Promedio de Clúster
        ax.plot(angles, cluster_avg_values, linewidth=2, linestyle='solid', color='#1f77b4', label=f'Promedio Clúster {assigned_cluster}')
        ax.fill(angles, cluster_avg_values, color='#1f77b4', alpha=0.3)

        # Graficar Valores de Usuario
        ax.plot(angles, user_values, linewidth=2, linestyle='solid', color='#d62728', label='Tu Resultado')
        ax.fill(angles, user_values, color='#d62728', alpha=0.15)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
