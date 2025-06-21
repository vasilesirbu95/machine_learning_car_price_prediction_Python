# 🚗 Machine Learning: Car Price Prediction

Ein Machine-Learning-Projekt zur Vorhersage von Autopreisen basierend auf verschiedenen Fahrzeugmerkmalen. Ziel ist es, einen möglichst präzisen Regressionsansatz zur Prognose realer Marktpreise zu entwickeln. Die Daten stammen von autoscout24 und sind für den Zeitraum von 2011 bis 2021 vorhanden.

## 🔍 Projektziele

- Entwicklung eines ML-Modells zur Preisprognose
- Vergleich mehrerer Regressionsverfahren (Lineare Regression, Polynomische Regression, Entscheidungsbaumregression, Random Forest Regression, Gradient Boosting Regression, Deep Learning)
- Evaluation der Modelle mittels gängiger Metriken (R²)
- GUI zur Ergebnisberechnung für unterschiedliche Marken und Modelle

## 📊 Verwendete Features (Auszug)

- Marke, Modell, Kilometerstand, Erstzulassung, Leistung

## 🧠 Genutzte Methoden & Tools

- **Python**, **pandas**, **NumPy**
- **TesorFlow**, **Keras** zum Erstellen des Deep Learning Modells
- **time** zur Bestimmung der Rechenzeit
- **joblib** zum Speichern der ML Modelle
- **os** zum Arbeiten mit Pfaden
- **Scikit-learn** für Modelltraining und -evaluierung
- **Matplotlib** für Visualisierung
- **PyQt5** zur Erstellung der GUI

## ⚙️ Workflow

1. Datenimport (kaggle API) & Exploration
2. Feature Engineering & Preprocessing
3. Modelltraining & Hyperparameter-Tuning
4. Evaluation & Visualisierung (GUI)

## 📈 Ergebnisse

- Bestes Modell: **Random Forest Regressor**
- Metrik: R²: 0.9267
- Hyperparameter: {'max_depth': 8,
                   'max_features': 'sqrt',
                   'min_samples_leaf': 1,
                   'min_samples_split': 4,
                   'n_estimators': 195}

*Hinweis: Ergebnisse abhängig vom genutzten Datensatz.*


## 📁 Projektstruktur

```bash
├── data/               # Eingabedaten
├── main/               # Hauptskript für Training & Modellierung
├── requirements.txt    # Python-Abhängigkeiten
└── README.md           # Diese Datei

Entwickler: Vasile Sirbu | Masterstudent Fahrzeugtechnik, TU Berlin | Schwerpunkt: Datenanalyse, Machine Learning, Automotive
Datum: 28.03.2025
