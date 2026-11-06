
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

st.title("🧠 Test de Personalidad Big Five (OCEAN)")

st.write(
    "Contesta las siguientes preguntas para conocer tus características "
    "de personalidad, obtener tu clúster y compararte con personas famosas."
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

# Verificar que existan las columnas necesarias
if not all(col in df.columns for col in features):
    st.error("El dataset no contiene todas las características Big Five.")
    st.stop()

# Asignar clústeres si el dataset no los contiene
if "cluster" not in df.columns:
    df_scaled = scaler.transform(df[features])
    df["cluster"] = kmeans.predict(df_scaled)

# Convertir etiquetas de clúster a números
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")

# Nombres de las características
nombres = {
    "op": "Apertura a la experiencia",
    "co": "Responsabilidad",
    "ex": "Extroversión",
    "ag": "Amabilidad",
    "ne": "Neuroticismo"
}

st.header("📝 Contesta las 5 preguntas")

st.write(
    "Selecciona qué tanto te identificas con cada afirmación. "
    "No hay respuestas correctas o incorrectas."
)

opciones = [
    "1 - Totalmente en desacuerdo",
    "2 - En desacuerdo",
    "3 - Neutral",
    "4 - De acuerdo",
    "5 - Totalmente de acuerdo"
]

with st.form("test_bigfive"):

    st.subheader("1. Apertura a la experiencia (OP)")
    op = st.radio(
        "Me gusta probar cosas nuevas y aprender sobre temas diferentes.",
        opciones,
        index=2
    )

    st.divider()

    st.subheader("2. Responsabilidad (CO)")
    co = st.radio(
        "Soy una persona organizada y cumplo con mis responsabilidades.",
        opciones,
        index=2
    )

    st.divider()

    st.subheader("3. Extroversión (EX)")
    ex = st.radio(
        "Disfruto convivir con otras personas y participar en actividades sociales.",
        opciones,
        index=2
    )

    st.divider()

    st.subheader("4. Amabilidad (AG)")
    ag = st.radio(
        "Me gusta ayudar a los demás y trabajar en equipo.",
        opciones,
        index=2
    )

    st.divider()

    st.subheader("5. Neuroticismo (NE)")
    ne = st.radio(
        "Con frecuencia me preocupo o me siento nervioso ante diferentes situaciones.",
        opciones,
        index=2
    )

    st.divider()

    enviar = st.form_submit_button(
        "🧠 Asignar mi clúster",
        type="primary",
        use_container_width=True
    )

if enviar:

    # Convertir respuestas de 1-5 a escala de 0-100
    def convertir_puntaje(respuesta):
        numero = int(respuesta.split(" - ")[0])
        return (numero - 1) * 25

    puntajes = {
        "op": convertir_puntaje(op),
        "co": convertir_puntaje(co),
        "ex": convertir_puntaje(ex),
        "ag": convertir_puntaje(ag),
        "ne": convertir_puntaje(ne)
    }

    input_data = pd.DataFrame([puntajes], columns=features)

    try:
        input_scaled = scaler.transform(input_data)
        assigned_cluster = kmeans.predict(input_scaled)[0]

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.stop()

    # Obtener los famosos del clúster
    famosos_cluster = df[df["cluster"] == assigned_cluster].copy()

    # Calcular promedio de características
    if famosos_cluster.empty:
        st.warning(
            "No se encontraron famosos registrados para este clúster."
        )
        st.stop()

    promedios = famosos_cluster[features].mean()

    st.divider()
    st.header("🎯 Resultado de tu personalidad")

    st.success(
        f"¡Tu personalidad ha sido asignada al CLÚSTER {assigned_cluster}!"
    )

    st.subheader("📌 Descripción de tu clúster")

    rasgos_ordenados = promedios.sort_values(ascending=False)

    nombres_cortos = {
        "op": "Apertura a la experiencia",
        "co": "Responsabilidad",
        "ex": "Extroversión",
        "ag": "Amabilidad",
        "ne": "Neuroticismo"
    }

    rasgo_mayor = nombres_cortos[rasgos_ordenados.index[0]]
    rasgo_menor = nombres_cortos[rasgos_ordenados.index[-1]]

    st.write(
        f"Este clúster se caracteriza principalmente por presentar "
        f"un nivel mayor de **{rasgo_mayor}** y un nivel menor de "
        f"**{rasgo_menor}**, de acuerdo con los promedios de sus integrantes."
    )

    st.write(
        "Estas características representan tendencias generales del grupo "
        "y no definen completamente la personalidad de una persona."
    )

    st.subheader("📊 Características predominantes del clúster")

    fig_bar, ax_bar = plt.subplots(figsize=(9, 5))

    valores = promedios.rename(index=nombres)

    valores.sort_values().plot(
        kind="barh",
        ax=ax_bar
    )

    ax_bar.set_title(
        f"Promedio de personalidad del Clúster {assigned_cluster}"
    )
    ax_bar.set_xlabel("Puntaje promedio (0 a 100)")
    ax_bar.set_ylabel("Características Big Five")
    ax_bar.set_xlim(0, 100)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig_bar)
    plt.close(fig_bar)


    st.subheader(
        f"👥 Top 5 de famosos del Clúster {assigned_cluster}"
    )

    if "usuario" in famosos_cluster.columns:

        columnas = ["usuario"]

        if "categoria" in famosos_cluster.columns:
            categoria_mapping = {
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
                famosos_cluster["categoria"].map(categoria_mapping)
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
        st.warning("No se encontró la columna 'usuario' en el dataset.")

    st.subheader("🕸️ Comparación de personalidad")

    st.write(
        "Compara tus resultados con el promedio de los famosos "
        "pertenecientes a tu clúster."
    )

    categorias = ["OP", "CO", "EX", "AG", "NE"]

    valores_usuario = [
        puntajes["op"],
        puntajes["co"],
        puntajes["ex"],
        puntajes["ag"],
        puntajes["ne"]
    ]

    valores_cluster = promedios[features].tolist()

    # Cerrar el radar
    valores_usuario += valores_usuario[:1]
    valores_cluster += valores_cluster[:1]

    N = len(categorias)

    angulos = [
        n / float(N) * 2 * np.pi
        for n in range(N)
    ]

    angulos += angulos[:1]

    fig_radar, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw=dict(polar=True)
    )

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
    ax.set_xticklabels(categorias)

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

    st.pyplot(fig_radar)
    plt.close(fig_radar)


    st.divider()

    st.caption(
        "Este resultado es una clasificación generada mediante "
        "un modelo de aprendizaje no supervisado K-Means. "
        "La clasificación es orientativa y no representa un diagnóstico psicológico."
    )

