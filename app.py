
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Test de Personalidad Big Five",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Clasificador de Personalidad Big Five")
st.write(
    "Descubre tu grupo de personalidad mediante el modelo "
    "de aprendizaje no supervisado K-Means."
)

@st.cache_data
def load_dataset():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    return pd.read_csv(url)

@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_BigFive_model.joblib")
    scaler = joblib.load("scaler_bigFive.joblib")
    return kmeans, scaler

try:
    df = load_dataset()
    kmeans, scaler = load_models()

except Exception as e:
    st.error(f"Error al cargar los recursos: {e}")
    st.stop()

features = ["op", "co", "ex", "ag", "ne"]

if not all(col in df.columns for col in features):
    st.error("Faltan columnas necesarias en el dataset.")
    st.stop()

# Asignar clústeres si no existen
if "cluster" not in df.columns:
    df_scaled = scaler.transform(df[features])
    df["cluster"] = kmeans.predict(df_scaled)

df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")

df = df.dropna(subset=["cluster"])

nombres = {
    "op": "Apertura a la experiencia",
    "co": "Responsabilidad",
    "ex": "Extroversión",
    "ag": "Amabilidad",
    "ne": "Neuroticismo"
}

col_preguntas, col_resultados = st.columns([1, 1.2], gap="large")

with col_preguntas:

    st.header("📝 Test de personalidad")

    st.write(
        "Indica qué tanto te identificas con cada afirmación."
    )

    st.caption(
        "0 = Totalmente en desacuerdo | 100 = Totalmente de acuerdo"
    )

    with st.form("test_bigfive"):

        st.markdown("**1. Apertura a la experiencia (OP)**")

        op_val = st.slider(
            "Me gusta probar cosas nuevas y aprender sobre temas diferentes.",
            min_value=0,
            max_value=100,
            value=50,
            key="op"
        )

        st.markdown("**2. Responsabilidad (CO)**")

        co_val = st.slider(
            "Soy una persona organizada y cumplo con mis responsabilidades.",
            min_value=0,
            max_value=100,
            value=50,
            key="co"
        )

        st.markdown("**3. Extroversión (EX)**")

        ex_val = st.slider(
            "Disfruto convivir con otras personas y participar en actividades sociales.",
            min_value=0,
            max_value=100,
            value=50,
            key="ex"
        )

        st.markdown("**4. Amabilidad (AG)**")

        ag_val = st.slider(
            "Me gusta ayudar a los demás y trabajar en equipo.",
            min_value=0,
            max_value=100,
            value=50,
            key="ag"
        )

        st.markdown("**5. Neuroticismo (NE)**")

        ne_val = st.slider(
            "Con frecuencia me preocupo o me siento nervioso ante diferentes situaciones.",
            min_value=0,
            max_value=100,
            value=50,
            key="ne"
        )

        enviar = st.form_submit_button(
            "🧠 Asignar mi clúster",
            type="primary",
            use_container_width=True
        )

if enviar:

    # Obtener los cinco valores
    input_data = pd.DataFrame([{
        "op": op_val,
        "co": co_val,
        "ex": ex_val,
        "ag": ag_val,
        "ne": ne_val
    }], columns=features)

    try:
        # Escalar los valores y realizar predicción
        input_scaled = scaler.transform(input_data)
        assigned_cluster = kmeans.predict(input_scaled)[0]

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.stop()

    # Obtener famosos del clúster
    famosos_cluster = df[df["cluster"] == assigned_cluster].copy()

    # Calcular promedios del clúster
    if famosos_cluster.empty:
        st.error(
            "No se encontraron personas en el dataset "
            "para el clúster asignado."
        )
        st.stop()

    promedios = famosos_cluster[features].mean()

    with col_resultados:

        st.header("🎯 Tu resultado")

        st.success(
            f"¡Perteneces al CLÚSTER {assigned_cluster}!"
        )

        st.subheader("📌 Descripción del clúster")

        # Identificar rasgos predominantes
        rasgos = promedios.sort_values(ascending=False)

        rasgo_mayor = nombres[rasgos.index[0]]
        rasgo_menor = nombres[rasgos.index[-1]]

        st.write(
            f"Este grupo presenta un promedio más alto en "
            f"**{rasgo_mayor}** y un promedio más bajo en "
            f"**{rasgo_menor}**."
        )

        st.write(
            "Estos resultados representan las características "
            "generales del grupo y no definen completamente "
            "la personalidad de cada persona."
        )

        st.subheader("📊 Características del clúster")

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))

        valores_barra = promedios.rename(index=nombres)
        valores_barra = valores_barra.sort_values()

        ax_bar.barh(
            valores_barra.index,
            valores_barra.values
        )

        ax_bar.set_xlabel("Puntaje promedio")
        ax_bar.set_title(
            f"Características del Clúster {assigned_cluster}"
        )
        ax_bar.set_xlim(0, 100)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.3)

        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)


    st.divider()

    st.header(
        f"👥 Top 5 de famosos del Clúster {assigned_cluster}"
    )

    if "usuario" in famosos_cluster.columns:

        columnas = ["usuario"]

        if "categoria" in famosos_cluster.columns:

            categorias = {
                1: "Actor/actriz",
                2: "Cantante",
                3: "Modelo",
                4: "TV y series",
                5: "Radio",
                6: "Tecnología",
                7: "Deportes",
                8: "Política",
                9: "Escritor"
            }

            famosos_cluster["categoria_nombre"] = (
                famosos_cluster["categoria"].map(categorias)
            )

            columnas.append("categoria_nombre")

        columnas += features

        top_5 = famosos_cluster[columnas].head(5).copy()

        top_5 = top_5.rename(columns={
            "usuario": "Famoso",
            "categoria_nombre": "Categoría",
            "op": "OP",
            "co": "CO",
            "ex": "EX",
            "ag": "AG",
            "ne": "NE"
        })

        st.dataframe(
            top_5,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "No se encontró la columna de famosos en el dataset."
        )


    st.divider()

    st.header("🕸️ Comparación de personalidad")

    st.write(
        "Compara tus resultados con el promedio de los famosos "
        "que pertenecen a tu mismo clúster."
    )

    categorias_radar = ["OP", "CO", "EX", "AG", "NE"]

    valores_usuario = [
        op_val,
        co_val,
        ex_val,
        ag_val,
        ne_val
    ]

    valores_cluster = promedios[features].tolist()

    # Cerrar las figuras
    valores_usuario += valores_usuario[:1]
    valores_cluster += valores_cluster[:1]

    N = len(categorias_radar)

    angulos = [
        n / float(N) * 2 * np.pi
        for n in range(N)
    ]

    angulos += angulos[:1]

    fig_radar, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw=dict(polar=True)
    )

    # Promedio del clúster
    ax.plot(
        angulos,
        valores_cluster,
        linewidth=2,
        label="Promedio del clúster"
    )

    ax.fill(
        angulos,
        valores_cluster,
        alpha=0.20
    )

    # Resultados del usuario
    ax.plot(
        angulos,
        valores_usuario,
        linewidth=2,
        label="Mis resultados"
    )

    ax.fill(
        angulos,
        valores_usuario,
        alpha=0.15
    )

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias_radar)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])

    ax.set_title(
        "Comparación Big Five",
        pad=25
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1)
    )

    st.pyplot(fig_radar, use_container_width=True)
    plt.close(fig_radar)

    st.divider()

    st.caption(
        "Clasificación realizada mediante el algoritmo "
        "K-Means de aprendizaje no supervisado. "
        "Los resultados son orientativos y no representan "
        "un diagnóstico psicológico."
    )

else:

    with col_resultados:
        st.header("🎯 Tu resultado aparecerá aquí")

        st.info(
            "Contesta las cinco preguntas y presiona "
            "'Asignar mi clúster' para conocer tu resultado."
        )
