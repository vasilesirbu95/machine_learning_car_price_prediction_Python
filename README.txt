Im folgenden Projekt werden die Preise der Fahrzeuge in Deutschland mit Hilfe von Machine Learning vorhergesagt.
Die Daten stammen von autoscout24 und sind für den Zeitraum von 2011 bis 2021 vorhanden.

Folgende Schritte wurden durchgeführt:
    1. Herunterladen des Datasets direkt von Kaggle mit Hilfe von API
        - Erstellen vom neuen Account bei Kaggle
        - Erstellen eines neuen Tokens --> Automatisches herunterladen von "kaggle.json"
        - Die JSON Datei in einem Ordner "C:Users/.kaggle" speichern
        - Daten als CSV herunterladen

    2. Einlesen, Bereinigen und Visualisieren der Daten
        - Die CSV-Datei einlesen
        - Entfernung von nicht relevanten/unnötigen Zeilen und Spalten
        - Löschen der Zeilen mit fehlenden Daten --> Alle Spalten besitzen die gleiche Länge
        - Dubletten entfernt
        - Zeilen mit unlogischen/fehlerhaften Daten gelöscht

    3. Machine Learning
        - Implementierung und Training der Linearen Regression für ein einziges Fahrzeugmodel
        - Berechnung der Prediction
        - Plotten der realen Daten als Scatter-Plot
        - Plotten der prädizierten Daten als Linien-Plot
        - Erweiterung der Linearen Regression über alle Fahrzeugmodelle
        - Anwendung von anderen Machine Learning Algorithmen
