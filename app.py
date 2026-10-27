"""ClimaAI · Simulador del data drift estacional (TP Final · Ejercicio 1).

La app reproduce, en vivo, el caso del ejercicio 1 y muestra ademas los
workflows de los ejercicios 2 y 3.

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).parent
DATASET = BASE / "dataset_consumo.csv"
DIAGRAMAS = BASE / "diagramas"
LOGO = BASE / "logo.jpg"

FEATURES = ["temperatura_media_c", "superficie_m2", "ocupantes", "es_fin_de_semana"]
NOMBRE_FEATURE = {
    "temperatura_media_c": "Temperatura media (°C)",
    "superficie_m2": "Superficie (m²)",
    "ocupantes": "Ocupantes",
    "es_fin_de_semana": "Fin de semana",
}
TEMPORADAS = ["verano", "otono", "invierno", "primavera"]
UMBRAL_RMSE = 2.5          # kWh/dia: por encima, la prediccion no sirve para operar
UMBRAL_PSI = 0.25          # deriva fuerte

C_AZUL, C_NARANJA, C_ROJO, C_VERDE = "#2563EB", "#B45309", "#B42318", "#0F766E"
C_INK, C_GRIS, C_LINEA = "#101828", "#5A6779", "#E4E9F0"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
    "axes.edgecolor": C_LINEA, "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_GRIS, "ytick.color": C_GRIS,
    "axes.grid": True, "grid.color": "#EEF1F6", "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 110,
})


# --------------------------------------------------------------------------- #
# Datos y modelos
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    return df


@st.cache_resource(show_spinner=False)
def entrenar(estaciones: tuple[str, ...], _datos: pd.DataFrame) -> RandomForestRegressor:
    """Entrena con las estaciones indicadas. La clave del ejercicio: mismo algoritmo."""
    sub = _datos[_datos["estacion"].isin(list(estaciones))]
    modelo = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0)
    modelo.fit(sub[FEATURES], sub["consumo_kwh"])
    return modelo


def metricas(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "R2": float(r2_score(y_real, y_pred)) if len(y_real) > 1 else float("nan"),
        "sesgo": float(np.mean(y_pred - y_real)),
        "n": int(len(y_real)),
    }


def psi(esperado: np.ndarray, real: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. 0 = distribuciones iguales; >0,25 = deriva fuerte."""
    esperado = np.asarray(esperado, dtype=float)
    real = np.asarray(real, dtype=float)
    if len(esperado) < 20 or len(real) < 20:
        return float("nan")
    cortes = np.quantile(esperado, np.linspace(0, 1, bins + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(esperado, bins=cortes)[0] / len(esperado)
    q = np.histogram(real, bins=cortes)[0] / len(real)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
def iniciar_estado() -> None:
    for clave, valor in [
        ("modelo_activo", "original"),
        ("stream", []),
        ("pool", None),
        ("pool_pos", 0),
        ("temporada_produccion", "invierno"),
    ]:
        st.session_state.setdefault(clave, valor)


def modelo_vigente(datos: pd.DataFrame) -> RandomForestRegressor:
    if st.session_state["modelo_activo"] == "corregido":
        return entrenar(tuple(TEMPORADAS), datos)
    return entrenar(("verano",), datos)


def reiniciar_stream() -> None:
    st.session_state["stream"] = []
    st.session_state["pool"] = None
    st.session_state["pool_pos"] = 0


def avanzar(datos: pd.DataFrame, n: int, temporada: str) -> None:
    """Genera n días de producción: predice con el modelo vigente y después revela la verdad."""
    if st.session_state["pool"] is None or st.session_state.get("pool_temporada") != temporada:
        pool = datos[datos["estacion"] == temporada].sample(frac=1.0, random_state=7)
        st.session_state["pool"] = pool.to_dict("records")
        st.session_state["pool_temporada"] = temporada
        st.session_state["pool_pos"] = 0
    pool = st.session_state["pool"]
    modelo = modelo_vigente(datos)
    for i in range(n):
        pos = st.session_state["pool_pos"]
        if pos >= len(pool):
            break
        fila = pool[pos]
        X = pd.DataFrame([{f: fila[f] for f in FEATURES}])
        pred = float(modelo.predict(X)[0])
        st.session_state["stream"].append({
            "n": len(st.session_state["stream"]) + 1,
            "fecha": str(fila["fecha"])[:10],
            "hogar": fila["hogar_id"],
            "temporada": fila["estacion"],
            "temperatura": float(fila["temperatura_media_c"]),
            "real": float(fila["consumo_kwh"]),
            "pred": pred,
            "modelo": st.session_state["modelo_activo"],
        })
        st.session_state["pool_pos"] = pos + 1


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def fig_distribucion(verano: np.ndarray, invierno: np.ndarray, valor_psi: float):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.linspace(-2, 40, 43)
    ax.hist(verano, bins=bins, color=C_AZUL, alpha=.75, label=f"Verano (entrenamiento) · media {verano.mean():.1f} °C",
            edgecolor="white", linewidth=.5)
    ax.hist(invierno, bins=bins, color=C_NARANJA, alpha=.75, label=f"Invierno (producción) · media {invierno.mean():.1f} °C",
            edgecolor="white", linewidth=.5)
    ax.axvline(verano.mean(), color=C_AZUL, ls="--", lw=1.8)
    ax.axvline(invierno.mean(), color=C_NARANJA, ls="--", lw=1.8)
    if not np.isnan(valor_psi):
        ax.set_title(f"Distribución de la temperatura · PSI = {valor_psi:.2f} "
                     f"({'deriva fuerte' if valor_psi > UMBRAL_PSI else 'sin deriva fuerte'})",
                     loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Temperatura media diaria (°C)")
    ax.set_ylabel("Días-hogar")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_real_vs_pred(y_real, y_pred, titulo: str, color: str = C_ROJO):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.scatter(y_real, y_pred, s=6, alpha=.20, color=color, edgecolors="none")
    lim = [0, float(max(np.max(y_real), np.max(y_pred))) * 1.05]
    ax.plot(lim, lim, ls="--", color=C_INK, lw=1.4, label="Predicción perfecta")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Consumo real (kWh)"); ax.set_ylabel("Consumo predicho (kWh)")
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def fig_serie(stream: list[dict]):
    df = pd.DataFrame(stream)
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(df["n"], df["real"], "o-", color=C_INK, lw=1.8, ms=3.5, label="Consumo real")
    for modelo, color, nombre in [("original", C_ROJO, "Modelo original (entrenado en verano)"),
                                  ("corregido", C_VERDE, "Modelo corregido (multi-estacional)")]:
        sub = df[df["modelo"] == modelo]
        if not sub.empty:
            ax.plot(sub["n"], sub["pred"], "s--", color=color, lw=1.6, ms=3.2, label=nombre)
    ax.set_xlabel("Día simulado de producción"); ax.set_ylabel("Consumo (kWh)")
    ax.set_title("Producción en vivo: predicción contra realidad", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_error_acumulado(stream: list[dict]):
    df = pd.DataFrame(stream)
    acumulado = []
    for i in range(1, len(df) + 1):
        m = metricas(df["real"].iloc[:i], df["pred"].iloc[:i])
        acumulado.append({"n": i, "MAE": m["MAE"], "RMSE": m["RMSE"]})
    acc = pd.DataFrame(acumulado)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(acc["n"], acc["MAE"], color=C_AZUL, lw=2, label="MAE")
    ax.plot(acc["n"], acc["RMSE"], color=C_ROJO, lw=2, label="RMSE")
    ax.axhline(UMBRAL_RMSE, color=C_GRIS, ls=":", lw=1.6, label=f"Umbral operativo ({UMBRAL_RMSE} kWh)")
    ax.set_xlabel("Día simulado de producción"); ax.set_ylabel("Error (kWh/día)")
    ax.set_title("Error acumulado del modelo vigente", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Diagramas
# --------------------------------------------------------------------------- #
def mostrar_svg(nombre: str, alto_min: int = 0) -> None:
    ruta = DIAGRAMAS / nombre
    if not ruta.exists():
        st.warning(f"Falta el diagrama {nombre} en la carpeta diagramas/.")
        return
    svg = ruta.read_text(encoding="utf-8")
    import re
    svg = re.sub(r'<svg([^>]*?)width="\d+" height="\d+"', r'<svg\1width="100%"', svg, count=1)
    estilo = f"min-height:{alto_min}px" if alto_min else ""
    st.markdown(
        f'<div style="background:#fff;border:1px solid {C_LINEA};border-radius:14px;padding:10px;{estilo}">'
        f'{svg}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Pestañas
# --------------------------------------------------------------------------- #
def tab_problema(datos: pd.DataFrame) -> None:
    st.subheader("El problema: el mismo modelo, dos estaciones")
    st.write(
        "Elegí con qué datos se entrena el modelo y sobre qué temporada se lo evalúa. "
        "El algoritmo y las variables son siempre los mismos: lo único que cambia es el conjunto de datos."
    )
    col1, col2 = st.columns(2)
    with col1:
        est_entrena = st.selectbox("Temporada de entrenamiento", TEMPORADAS, index=0,
                                   help="El modelo original de ClimaAI se entrenó solo con el verano.")
    with col2:
        est_evalua = st.selectbox("Temporada de producción (evaluación)", TEMPORADAS, index=2,
                                  help="El invierno es donde apareció el fallo reportado por operaciones.")

    if st.button("Entrenar y evaluar", type="primary"):
        modelo = entrenar((est_entrena,), datos)
        test = datos[datos["estacion"] == est_evalua]
        pred = modelo.predict(test[FEATURES])
        st.session_state["resultado"] = {
            "entrena": est_entrena,
            "evalua": est_evalua,
            "metricas": metricas(test["consumo_kwh"], pred),
            "psi": psi(datos.loc[datos["estacion"] == est_entrena, "temperatura_media_c"].values,
                       test["temperatura_media_c"].values),
            "real": test["consumo_kwh"].to_numpy(),
            "pred": pred,
            "temp_entrena": datos.loc[datos["estacion"] == est_entrena, "temperatura_media_c"].to_numpy(),
            "temp_evalua": test["temperatura_media_c"].to_numpy(),
        }

    res = st.session_state.get("resultado")
    if res is None:
        st.info("Todavía no entrenaste ningún modelo. El botón de arriba entrena y evalúa en un par de segundos.")
        return

    m = res["metricas"]
    mismo = res["entrena"] == res["evalua"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{m['MAE']:.2f} kWh")
    c2.metric("RMSE", f"{m['RMSE']:.2f} kWh")
    c3.metric("R²", f"{m['R2']:.2f}")
    c4.metric("PSI (deriva)", "—" if np.isnan(res["psi"]) else f"{res['psi']:.2f}")

    if mismo:
        st.success(f"Entrenado y evaluado en {res['entrena']}: el modelo está en su dominio y el error es bajo.")
    elif m["RMSE"] > UMBRAL_RMSE:
        st.error(
            f"Entrenado en {res['entrena']} y evaluado en {res['evalua']}: el RMSE de {m['RMSE']:.2f} kWh "
            f"está por encima del umbral operativo de {UMBRAL_RMSE} kWh y el sesgo medio es "
            f"{m['sesgo']:+.2f} kWh (predice de menos). Esto es data drift, no un error de código."
        )
    else:
        st.warning("El error está contenido, pero conviene mirar la deriva de las variables de entrada.")

    if not np.isnan(res["psi"]) and res["psi"] > UMBRAL_PSI:
        st.caption(
            f"El PSI de la temperatura entre {res['entrena']} y {res['evalua']} es {res['psi']:.2f}. "
            f"Por encima de {UMBRAL_PSI} se considera deriva fuerte de la distribución de entrada."
        )

    izq, der = st.columns(2)
    with izq:
        st.pyplot(fig_distribucion(res["temp_entrena"], res["temp_evalua"], res["psi"]), width="stretch")
    with der:
        color = C_VERDE if mismo or m["RMSE"] <= UMBRAL_RMSE else C_ROJO
        st.pyplot(fig_real_vs_pred(res["real"], res["pred"],
                                   f"Predicho contra real en {res['evalua']}", color), width="stretch")


def tab_correccion(datos: pd.DataFrame) -> None:
    st.subheader("La corrección: mismo algoritmo, dataset completo")
    st.write(
        "La consigna pide arreglarlo dentro del ciclo de vida del dato, sin tocar la arquitectura algorítmica "
        "ni la infraestructura. Eso es exactamente lo que hace este botón: vuelve a entrenar el mismo "
        "Random Forest, con las mismas variables, sobre el dataset de las cuatro estaciones."
    )
    if st.button("Re-entrenar con el dataset multi-estacional", type="primary"):
        verano = datos[datos["estacion"] == "verano"]
        invierno = datos[datos["estacion"] == "invierno"]
        original = entrenar(("verano",), datos)
        corregido = entrenar(tuple(TEMPORADAS), datos)
        st.session_state["comparacion"] = {
            "A_verano": metricas(verano["consumo_kwh"], original.predict(verano[FEATURES])),
            "A_invierno": metricas(invierno["consumo_kwh"], original.predict(invierno[FEATURES])),
            "C_invierno": metricas(invierno["consumo_kwh"], corregido.predict(invierno[FEATURES])),
        }

    comp = st.session_state.get("comparacion")
    if comp is None:
        st.info("Ejecutá el re-entrenamiento para ver la comparación antes/después.")
        return

    tabla = pd.DataFrame([
        {"Configuración": "Original · entrenado en verano · medido en verano",
         "MAE (kWh)": round(comp["A_verano"]["MAE"], 2),
         "RMSE (kWh)": round(comp["A_verano"]["RMSE"], 2),
         "R²": round(comp["A_verano"]["R2"], 2)},
        {"Configuración": "Original · entrenado en verano · medido en invierno",
         "MAE (kWh)": round(comp["A_invierno"]["MAE"], 2),
         "RMSE (kWh)": round(comp["A_invierno"]["RMSE"], 2),
         "R²": round(comp["A_invierno"]["R2"], 2)},
        {"Configuración": "Corregido · multi-estacional · medido en invierno",
         "MAE (kWh)": round(comp["C_invierno"]["MAE"], 2),
         "RMSE (kWh)": round(comp["C_invierno"]["RMSE"], 2),
         "R²": round(comp["C_invierno"]["R2"], 2)},
    ])
    st.dataframe(tabla, width="stretch", hide_index=True)

    antes, despues = comp["A_invierno"], comp["C_invierno"]
    st.success(
        f"En invierno el error promedio pasa de {antes['MAE']:.2f} kWh a {despues['MAE']:.2f} kWh, "
        f"y el R² de {antes['R2']:.2f} a {despues['R2']:.2f}. El modelo vuelve a servir para operar."
    )
    st.caption(
        "En un sistema real este paso viene después de re-ingestar y etiquetar datos de la estación nueva, "
        "validar su calidad y recién entonces volver a entrenar."
    )


def tab_produccion(datos: pd.DataFrame) -> None:
    st.subheader("Simulador de producción en vivo")
    st.write(
        "Cada día que se genera es un hogar en una fecha concreta: el modelo predice antes de conocer el "
        "valor real y después se revela. Las métricas se acumulan como se acumularían en un sistema en "
        "operación. Cambiá la temporada para ver cómo se comporta el modelo en cada estación."
    )
    col1, col2, col3 = st.columns([2, 1.2, 1.2])
    with col1:
        temporada = st.selectbox("Temporada que llega del mundo real", TEMPORADAS,
                                 index=TEMPORADAS.index(st.session_state["temporada_produccion"]))
        st.session_state["temporada_produccion"] = temporada
    with col2:
        paso = st.selectbox("Cuántos días agregar", [1, 10, 25], index=1)
    with col3:
        st.write("")
        generar = st.button("Generar días", type="primary")

    col_a, col_b, col_c = st.columns([1.4, 1.6, 1.6])
    with col_a:
        if generar:
            avanzar(datos, paso, temporada)
        if st.button("Aplicar corrección"):
            st.session_state["modelo_activo"] = "corregido"
        if st.button("Reiniciar simulación"):
            reiniciar_stream()
    with col_b:
        etiqueta = "corregido (multi-estacional)" if st.session_state["modelo_activo"] == "corregido" else "original (entrenado en verano)"
        st.info(f"Modelo vigente en producción: {etiqueta}")
    with col_c:
        st.metric("Días simulados", len(st.session_state["stream"]))

    stream = st.session_state["stream"]
    if not stream:
        st.info("Todavía no hay días simulados. Usá el botón «Generar días».")
        return

    df = pd.DataFrame(stream)
    m = metricas(df["real"], df["pred"])
    psi_stream = psi(datos.loc[datos["estacion"] == "verano", "temperatura_media_c"].values,
                     df["temperatura"].values)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE acumulado", f"{m['MAE']:.2f} kWh")
    c2.metric("RMSE acumulado", f"{m['RMSE']:.2f} kWh")
    c3.metric("R² acumulado", f"{m['R2']:.2f}")
    c4.metric("PSI acumulado", "—" if np.isnan(psi_stream) else f"{psi_stream:.2f}")

    if m["RMSE"] > UMBRAL_RMSE:
        st.error(
            f"Alerta: el RMSE acumulado ({m['RMSE']:.2f} kWh) supera el umbral operativo de {UMBRAL_RMSE} kWh. "
            f"El modelo está prediciendo de menos en promedio {abs(m['sesgo']):.2f} kWh por día. "
            f"Corresponde revisar la deriva de las variables de entrada y volver a entrenar con datos de la "
            f"estación nueva."
        )
    elif m["RMSE"] > 1.5:
        st.warning("El error está creciendo. Conviene mirar la deriva antes de que el sistema falle.")
    else:
        st.success("El modelo vigente está dentro del umbral operativo.")

    st.pyplot(fig_serie(stream), width="stretch")
    st.pyplot(fig_error_acumulado(stream), width="stretch")

    with st.expander(f"Ver los últimos días simulados ({min(len(df), 15)} de {len(df)})"):
        st.dataframe(df.tail(15), width="stretch", hide_index=True)


def tab_ejercicio2() -> None:
    st.subheader("Ejercicio 2 · Memoria conversacional y arquitectura RAG")
    st.write(
        "El agente de TaxAgent inyectaba el manual impositivo completo (5.000 páginas) en cada mensaje. "
        "El costo se volvió insostenible y el historial de la conversación quedó fuera de la ventana de "
        "contexto, lo que produjo el olvido a los tres mensajes."
    )
    izq, der = st.columns(2)
    with izq:
        st.markdown("**Qué falló**")
        st.markdown(
            "- El modelo es stateless: cada llamada es independiente y solo conoce lo que viaja en el prompt.\n"
            "- No existía un componente que persistiera el historial por sesión.\n"
            "- El manual ocupaba casi toda la ventana de contexto, así que no había lugar para la conversación."
        )
    with der:
        st.markdown("**Cómo se corrige**")
        st.markdown(
            "- Memoria de sesión persistente, identificada por `session_id`.\n"
            "- Ventana deslizante: solo los últimos N turnos entran al prompt.\n"
            "- Resumen acumulativo para los turnos viejos.\n"
            "- Almacén vectorial para los hechos estables del caso."
        )
    st.markdown("#### El workflow, antes y después")
    mostrar_svg("n8n_ej2_antes_despues.svg")
    st.markdown("#### El pipeline de recuperación, paso a paso")
    mostrar_svg("flujo_ej2_rag.svg")
    st.caption(
        "Arriba, la ruta offline que indexa el manual una sola vez. Abajo, la ruta online que se ejecuta en "
        "cada consulta y que reemplaza a la inyección masiva de texto."
    )


def tab_ejercicio3() -> None:
    st.subheader("Ejercicio 3 · Control humano y trazabilidad (ISO/IEC 42001)")
    st.write(
        "Un agente ReAct con permisos para ejecutar scripts fue engañado con una falsa alerta de seguridad "
        "nacional y borró directorios operativos. La corrección no pasa por blindar el prompt, que no se puede "
        "demostrar completo, sino por poner un freno humano delante de las acciones irreversibles y registrar "
        "cada decisión."
    )
    izq, der = st.columns([1.2, 1])
    with izq:
        st.markdown("**El freno: Human-in-the-Loop**")
        st.markdown(
            "- Toda acción destructiva queda en estado `pending`: no se ejecuta sola.\n"
            "- La política por defecto es negar (fail-safe).\n"
            "- La persona verifica fuera de banda lo que el modelo no puede: origen, autoridad, contexto.\n"
            "- Cada acción crítica queda asociada a quien la aprobó."
        )
    with der:
        st.markdown("**La evidencia: registro de auditoría**")
        st.json({
            "session_id": "sess-global-20261022-8841",
            "timestamp": "2026-10-22T21:12:54-03:00",
            "tool_invoked": "shell.execute — rm -rf /ordenes/activas",
            "decision": "Bloqueada: comando destructivo clasificado crítico",
            "human_approval": {
                "required": True,
                "status": "rejected",
                "approver": "m.perez@globaldeliver.com",
            },
        })
    st.markdown("#### El workflow, antes y después")
    mostrar_svg("n8n_ej3_antes_despues.svg")
    st.markdown("#### El flujo con el gate de aprobación")
    mostrar_svg("flujo_ej3_hitl.svg")


def tab_datos(datos: pd.DataFrame) -> None:
    st.subheader("Métricas y dataset")
    st.markdown(
        "Las tres métricas de regresión que se usan para monitorear el error, con la razón de cada una, y una "
        "vista del dataset sintético con el que se reproducen los resultados del documento."
    )
    mostrar_svg("infografia_metricas.svg")

    st.markdown("#### Resumen del dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(datos):,}".replace(",", "."))
    c2.metric("Hogares", datos["hogar_id"].nunique())
    c3.metric("Período", f"{datos['fecha'].min():%Y-%m-%d} a {datos['fecha'].max():%Y-%m-%d}")
    c4.metric("Consumo medio", f"{datos['consumo_kwh'].mean():.1f} kWh/día")

    resumen = (datos.groupby("estacion")["consumo_kwh"]
               .agg(dias="count", consumo_medio="mean", temperatura_media="mean")
               .round(2).reset_index())
    st.dataframe(resumen, width="stretch", hide_index=True)

    with st.expander("Ver las primeras filas del dataset"):
        st.dataframe(datos.head(50), width="stretch", hide_index=True)
    with st.expander("Cómo se generó el dataset"):
        st.markdown(
            "Son 120 hogares durante los 365 días de 2025, con el calendario del hemisferio sur. El consumo se "
            "genera con reglas físicas: carga base por superficie y ocupantes, más horas de aire acondicionado "
            "cuando la temperatura supera los 24 °C, más horas de calefacción cuando baja de 14 °C, más una "
            "carga de iluminación en invierno y un ruido irreducible. La semilla es fija, así que los "
            "resultados se reproducen. El generador es `build_dataset.py`."
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="ClimaAI · Simulador de data drift",
        page_icon=str(LOGO) if LOGO.exists() else ":material/thermostat:",
        layout="wide",
    )
    iniciar_estado()
    datos = cargar_datos()

    with st.sidebar:
        if LOGO.exists():
            st.image(str(LOGO), width=120)
        st.markdown("### ClimaAI · simulador del Ejercicio 1")
        st.caption("Trabajo Práctico Final · Experto en Arquitectura y Desarrollo de IA")
        st.markdown(
            "Un modelo de regresión que estima el consumo eléctrico diario funciona en verano y se derrumba en "
            "invierno. Acá se puede reproducir el fallo, corregirlo y verlo en vivo."
        )
        st.divider()
        st.markdown("**Ficha técnica**")
        st.caption(
            f"Algoritmo: Random Forest · {len(FEATURES)} variables de entrada\n\n"
            f"Variables: {', '.join(NOMBRE_FEATURE[f] for f in FEATURES)}\n\n"
            f"Objetivo: consumo_kwh (kWh/día)\n\n"
            f"Umbral operativo de RMSE: {UMBRAL_RMSE} kWh\n\n"
            f"Umbral de PSI: {UMBRAL_PSI}"
        )

    st.title("Diagnóstico, corrección y monitoreo del data drift")
    tabs = st.tabs([
        "1 · El problema",
        "2 · La corrección",
        "3 · Producción en vivo",
        "Ejercicio 2 · RAG",
        "Ejercicio 3 · HITL",
        "Métricas y dataset",
    ])
    with tabs[0]:
        tab_problema(datos)
    with tabs[1]:
        tab_correccion(datos)
    with tabs[2]:
        tab_produccion(datos)
    with tabs[3]:
        tab_ejercicio2()
    with tabs[4]:
        tab_ejercicio3()
    with tabs[5]:
        tab_datos(datos)


if __name__ == "__main__":
    main()
