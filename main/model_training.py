"""Train and persist the best price model for every make/model combination."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


DEFAULT_FEATURES = ["Kilometerstand", "Leistung", "Baujahr"]
DEFAULT_TARGET = "Preis"


def model_file_path(output_dir: str | Path, brand: str, model: str) -> Path:
    """Return a collision-safe and filesystem-safe path for one vehicle model."""
    safe_brand = re.sub(r"[^A-Za-z0-9_-]+", "_", str(brand)).strip("_")
    safe_model = re.sub(r"[^A-Za-z0-9_-]+", "_", str(model)).strip("_")
    return Path(output_dir) / f"{safe_brand}_{safe_model}_ml.joblib"


def legacy_model_file_path(output_dir: str | Path, brand: str, model: str) -> Path:
    """Return the filename used by the original training notebook."""
    safe_brand = re.sub(r"[^A-Za-z0-9_-]+", "_", str(brand)).strip("_")
    safe_model = re.sub(r"[^A-Za-z0-9_-]+", "_", str(model)).strip("_")
    return Path(output_dir) / f"{safe_brand}_{safe_model[:2]}_ml.joblib"


def find_model_file(output_dir: str | Path, brand: str, model: str) -> Path | None:
    """Find a current model first, then fall back to the legacy filename."""
    current_path = model_file_path(output_dir, brand, model)
    if current_path.exists():
        return current_path
    legacy_path = legacy_model_file_path(output_dir, brand, model)
    if legacy_path.exists():
        return legacy_path

    # Some early notebook runs kept the space in a two-character model prefix,
    # e.g. ``A `` in ``Mercedes-Benz_A _ml.joblib``.
    safe_brand = re.sub(r"[^A-Za-z0-9_-]+", "_", str(brand)).strip("_")
    historical_prefix = str(model)[:2]
    historical_path = Path(output_dir) / f"{safe_brand}_{historical_prefix}_ml.joblib"
    return historical_path if historical_path.exists() else None


def cleanup_saved_models(
    df: pd.DataFrame,
    output_dir: str | Path = "car_price_models",
    min_samples: int = 50,
    min_r2: float = 0.6,
) -> pd.DataFrame:
    """Delete saved models without enough data or with an insufficient R²."""
    output_path = Path(output_dir)
    counts = df.groupby(["Marke", "Model"]).size()
    report: list[dict[str, Any]] = []

    for model_path in output_path.glob("*.joblib"):
        reason = ""
        try:
            saved_model = joblib.load(model_path)
        except (OSError, EOFError, ImportError, ValueError, AttributeError) as error:
            reason = f"ungültige Modelldatei: {error}"
        else:
            score = getattr(saved_model, "best_score_", None)
            if score is None or score < min_r2:
                reason = f"R² {score!r} < {min_r2}"

            stem = model_path.stem.removesuffix("_ml")
            brand, _, model_prefix = stem.rpartition("_")
            candidates = counts[
                [
                    brand_value == brand
                    and str(model_value).startswith(model_prefix)
                    and count >= min_samples
                    for (brand_value, model_value), count in counts.items()
                ]
            ]
            if candidates.empty:
                reason = "keine ausreichenden Fahrzeugdaten"

        if reason:
            model_path.unlink()
            report.append({"datei": str(model_path), "grund": reason})

    return pd.DataFrame(report)


def _model_searches() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Create fresh estimators and grids for one vehicle-model search."""
    return {
        "Lineare Regression": (
            LinearRegression(),
            {"fit_intercept": [True, False]},
        ),
        "Polynomische Regression": (
            Pipeline(
                [
                    ("polynomial", PolynomialFeatures(include_bias=False)),
                    ("regression", LinearRegression()),
                ]
            ),
            {
                "polynomial__degree": [2, 3],
                "regression__fit_intercept": [True, False],
            },
        ),
        "Entscheidungsbaumregression": (
            DecisionTreeRegressor(random_state=42),
            {
                "max_depth": [7, 8, 9],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2],
            },
        ),
        "Random Forest Regression": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [7, 9, None],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 1.0],
            },
        ),
        "Gradient Boosting Regression": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }


def compare_models_for_vehicle(
    vehicle_df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[str, Any, pd.DataFrame]:
    """Run an independent grid search for each candidate and select by test R²."""
    features = features or DEFAULT_FEATURES
    X_train, X_test, y_train, y_test = train_test_split(
        vehicle_df[features],
        vehicle_df[target],
        test_size=test_size,
        random_state=random_state,
    )

    results: list[dict[str, Any]] = []
    best_name = ""
    best_estimator = None
    best_score = float("-inf")

    for name, (estimator, parameter_grid) in _model_searches().items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        overall_score = r2_score(
            vehicle_df[target], search.best_estimator_.predict(vehicle_df[features])
        )
        results.append(
            {
                "Modell": name,
                "R2": overall_score,
                "Parameter": search.best_params_,
            }
        )
        if overall_score > best_score:
            best_name = name
            best_estimator = search.best_estimator_
            best_score = overall_score

    if best_estimator is None:
        raise RuntimeError("Es konnte kein Modell trainiert werden.")

    return best_name, best_estimator, pd.DataFrame(results).sort_values(
        "R2", ascending=False
    )


def train_and_save_models(
    df: pd.DataFrame,
    output_dir: str | Path = "car_price_models",
    features: list[str] | None = None,
    target: str = DEFAULT_TARGET,
    min_samples: int = 50,
) -> pd.DataFrame:
    """Train, compare and save one best estimator for every make/model pair."""
    features = features or DEFAULT_FEATURES
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    comparisons: list[pd.DataFrame] = []

    for (brand, model), vehicle_df in df.groupby(["Marke", "Model"]):
        if len(vehicle_df) < min_samples:
            continue
        best_name, best_estimator, comparison = compare_models_for_vehicle(
            vehicle_df, features=features, target=target
        )
        comparison = comparison.assign(
            Marke=brand,
            Model=model,
            ausgewaehlt=comparison["Modell"].eq(best_name),
        )
        comparisons.append(comparison)
        destination = model_file_path(output_path, brand, model)
        joblib.dump(best_estimator, destination)
        best_row = comparison.iloc[0]
        summaries.append(
            {
                "Marke": brand,
                "Model": model,
                "bestes_Modell": best_name,
                "R2": best_row["R2"],
                "datei": str(destination),
            }
        )

    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(
            output_path / "model_comparison_results.csv", index=False
        )
    return pd.DataFrame(summaries).sort_values(["Marke", "Model"]).reset_index(drop=True)
