"""Streamlit interface for vehicle price prediction."""

import ast
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import r2_score

try:
    from main.model_training import find_model_file
except ModuleNotFoundError:
    from model_training import find_model_file

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "autoscout24-germany-dataset_cleaned.csv"
MODEL_DIR = ROOT_DIR / "car_price_models"
FEATURES = ["Kilometerstand", "Leistung", "Baujahr"]

st.set_page_config(page_title="AutoWert | Preisprognose", page_icon="🚗", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; }
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .hero { background: linear-gradient(120deg,#102a43,#1677b8); color:white;
            border-radius:22px; padding:2.2rem 2.4rem; margin-bottom:1.5rem;
            box-shadow:0 10px 28px rgba(16,42,67,.16); }
    .hero h1 { font-size:2.5rem; margin:0 0 .35rem; }
    .hero p { color:#d9efff; font-size:1.08rem; margin:0; }
    .result-card { background:white; border:1px solid #dce6ef; border-radius:18px;
                   padding:1.5rem; box-shadow:0 8px 22px rgba(16,42,67,.08); }
    .price { color:#0b7a75; font-size:2.55rem; font-weight:750; }
    .muted { color:#627d98; font-size:.92rem; }
    div[data-testid="stFormSubmitButton"] button { background:#0b7a75; color:white;
        border:0; min-height:3rem; font-weight:650; width:100%; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_vehicle_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Datensatz nicht gefunden: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    path = MODEL_DIR / "model_comparison_results.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_vehicle_image(brand: str, model: str) -> dict[str, str] | None:
    """Find one representative production vehicle image per make and model."""
    excluded_terms = {
        "race", "racing", "rally", "dtm", "concept", "prototype",
        "render", "formula", "nascar", "wrc", "le mans", "museum",
        "toy", "spielzeug", "model", "miniature", "badge", "emblem",
        "logo", "sign", "diecast", "schild", "plakette",
    }
    candidates = []
    search_phrase = f"{brand} {model}".lower()
    queries = [(f'"{brand} {model}"', search_phrase)]
    if str(model).isdigit() and len(str(model)) == 3:
        series = f"{brand} {str(model)[0]} series".lower()
        queries.append((f'"{brand} {str(model)[0]} series" car', series))
    for query, required_phrase in queries:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": 20,
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": 900,
                "format": "json",
            },
            headers={"User-Agent": "AutoWert/1.0 vehicle price prediction app"},
            timeout=8,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        required_tokens = re.findall(r"[a-z0-9]+", required_phrase)
        for page in pages.values():
            title = page.get("title", "").lower().removeprefix("file:")
            if any(term in title for term in excluded_terms):
                continue
            normalized_title = re.sub(r"[^a-z0-9]+", " ", title)
            if not all(token in normalized_title.split() for token in required_tokens):
                continue
            historical_years = re.findall(r"\b(18\d{2}|19\d{2})\b", title)
            if historical_years:
                continue
            has_car_indicators = any(
                term in normalized_title
                for term in ["car", "auto", "automobile", "sedan", "coupe", "hatchback", "cabriolet", "suv", "van", "kombi", "estate"]
            )
            if not has_car_indicators:
                continue
            image_info = page.get("imageinfo", [{}])[0]
            image_url = image_info.get("thumburl") or image_info.get("url")
            if not image_url:
                continue
            score = int(search_phrase in title) * 10 + int(required_phrase in title)
            candidates.append((score, title, image_url, image_info.get("descriptionurl")))
        if candidates:
            break
    if not candidates:
        return None
    score, title, image_url, source_url = max(candidates, key=lambda item: item[0])
    return {"url": image_url, "source_url": source_url or "", "title": title}


def format_euro(value: float) -> str:
    return f"{value:,.0f} €".replace(",", ".")


try:
    vehicle_data = load_vehicle_data()
except (FileNotFoundError, pd.errors.ParserError) as error:
    st.error(f"Die Fahrzeugdaten konnten nicht geladen werden: {error}")
    st.stop()

comparison = load_model_comparison()
if comparison is not None and "R2" not in comparison.columns:
    comparison = comparison.rename(columns={"R2_test": "R2"})

model_scores = {}
if comparison is not None and {"Marke", "Model", "R2"}.issubset(comparison.columns):
    model_scores = (
        comparison.groupby(["Marke", "Model"])["R2"].max().to_dict()
    )

supported_pairs = []
for (candidate_brand, candidate_model), candidate_data in vehicle_data.groupby(["Marke", "Model"]):
    if len(candidate_data) < 50:
        continue
    candidate_path = find_model_file(MODEL_DIR, candidate_brand, candidate_model)
    if candidate_path is None:
        continue
    try:
        load_model(str(candidate_path))
    except (OSError, ValueError, ImportError):
        continue
    candidate_score = model_scores.get((candidate_brand, candidate_model))
    if candidate_score is not None and candidate_score > 0.6:
        supported_pairs.append((candidate_brand, candidate_model))

if not supported_pairs:
    st.error("Es sind keine Fahrzeugmodelle mit mindestens 50 Datensätzen und R² > 0,6 verfügbar.")
    st.stop()

supported_data = vehicle_data[
    vehicle_data.set_index(["Marke", "Model"]).index.isin(supported_pairs)
].copy()

st.markdown(
    """
    <div class="hero">
        <h1>🚗 AutoWert</h1>
        <p>Intelligente Preisprognose für Gebrauchtwagen – schnell, transparent und modellgenau.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

brands = sorted(supported_data["Marke"].dropna().unique())
st.sidebar.header("Fahrzeug auswählen")
brand = st.sidebar.selectbox("Marke", brands)
models = sorted(supported_data.loc[supported_data["Marke"].eq(brand), "Model"].dropna().unique())
model = st.sidebar.selectbox("Modell", models)

with st.sidebar.form("prediction_form"):
    st.caption("Fahrzeugdaten")
    mileage = st.number_input("Kilometerstand", min_value=0, max_value=1_000_000, value=100_000, step=5_000)
    power = st.number_input("Leistung (PS)", min_value=1, max_value=2_000, value=150, step=5)
    year = st.number_input("Baujahr", min_value=1950, max_value=2030, value=2018, step=1)
    submitted = st.form_submit_button("Preis berechnen")

selected_comparison = None
if comparison is not None:
    selected_comparison = comparison[
        comparison["Marke"].eq(brand) & comparison["Model"].eq(model)
    ].sort_values("R2", ascending=False)

price_tab, analysis_tab = st.tabs(["Preisprognose", "Datensatz & Modellanalyse"])

with price_tab:
    st.subheader(f"{brand} {model}")
    st.markdown(
        '<p class="muted">Wähle links ein Fahrzeugmodell und ergänze die drei Fahrzeugdaten.</p>',
        unsafe_allow_html=True,
    )

    if submitted:
        model_path = find_model_file(MODEL_DIR, brand, model)
        if model_path is None:
            st.error(
                "Für diese Auswahl wurde noch kein Modell gespeichert. "
                "Bitte führe zunächst das Training aus."
            )
        else:
            try:
                estimator = load_model(str(model_path))
                prediction = float(
                    estimator.predict(
                        pd.DataFrame([[mileage, power, year]], columns=FEATURES)
                    )[0]
                )
            except (OSError, ValueError, KeyError) as error:
                st.error(f"Die Preisprognose konnte nicht berechnet werden: {error}")
            else:
                price_column, image_column = st.columns([1, 1])
                with price_column:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="muted">Geschätzter Marktpreis</div>
                            <div class="price">{format_euro(max(0, prediction))}</div>
                            <div class="muted">Individuelles ML-Modell für {brand} {model}.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with image_column:
                    try:
                        vehicle_image = load_vehicle_image(brand, model)
                    except (requests.RequestException, ValueError):
                        vehicle_image = None
                    if vehicle_image:
                        st.image(
                            vehicle_image["url"],
                            caption=f"{brand} {model}",
                        )
                        if vehicle_image["source_url"]:
                            st.caption(f"[Bildquelle auf Wikimedia Commons]({vehicle_image['source_url']})")
                    else:
                        st.info(f"Kein passendes Serienfahrzeugbild für {brand} {model} gefunden.")
    else:
        st.info("Trage die Fahrzeugdaten links ein und klicke auf „Preis berechnen“.")

with analysis_tab:
    st.subheader(f"Analyse: {brand} {model}")
    selected_data = vehicle_data[
        (vehicle_data["Marke"] == brand) & (vehicle_data["Model"] == model)
    ].copy()

    if selected_data.empty:
        st.warning("Für diese Auswahl sind keine Datensätze vorhanden.")
    else:
        st.metric("Datensätze", f"{len(selected_data):,}".replace(",", "."))
        st.dataframe(selected_data, use_container_width=True, hide_index=True)
        model_path = find_model_file(MODEL_DIR, brand, model)
        if model_path is None:
            st.info("Für dieses Fahrzeugmodell wurde noch keine Modell-Datei gespeichert.")
        else:
            try:
                estimator = load_model(str(model_path))
                selected_estimator = getattr(estimator, "best_estimator_", estimator)
                best_name = type(selected_estimator).__name__
                actual = selected_data["Preis"].astype(float)
                predicted = estimator.predict(selected_data[FEATURES])
                r2_value = r2_score(actual, predicted)
                if selected_comparison is not None and not selected_comparison.empty:
                    best_row = selected_comparison.iloc[0]
                    parameters = ast.literal_eval(str(best_row["Parameter"]))
                else:
                    # Ältere Trainingsläufe haben noch keine Vergleichsdatei gespeichert.
                    parameters = getattr(estimator, "best_params_", selected_estimator.get_params())

                st.subheader("Optimales ML-Modell")
                metric_columns = st.columns(2)
                metric_columns[0].metric("Verfahren", best_name)
                metric_columns[1].metric("R²", f"{r2_value:.3f}")

                st.subheader("Optimierte Parameter")
                st.json(parameters)

                figure, axis = plt.subplots(figsize=(9, 5))
                axis.scatter(actual, predicted, alpha=0.45, color="#1677b8")
                limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
                axis.plot(limits, limits, "--", color="#d64545", label="Ideale Vorhersage")
                axis.set_title("Modellkurve: tatsächliche vs. vorhergesagte Preise")
                axis.set_xlabel("Tatsächlicher Preis (€)")
                axis.set_ylabel("Vorhergesagter Preis (€)")
                axis.grid(alpha=0.2)
                axis.legend()
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)
            except (OSError, ValueError, KeyError, AttributeError) as error:
                st.warning(f"Die Modellanalyse konnte nicht erstellt werden: {error}")
