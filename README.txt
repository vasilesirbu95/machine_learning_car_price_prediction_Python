Im folgenden Projekt werden die Preise der Fahrzeuge in Deutschland mit Hilfe von Machine Learning vorhergesagt.
Die Daten stammen von autoscout24 und sind für den Zeitraum von 2011 bis 2021 vorhanden.

Folgende Schritte wurden durchgeführt:
    1. Herunterladen des Datasets direkt von Kaggle mit Hilfe von API
        - Erstellen vom neuen Account bei Kaggle
        - Erstellen eines neuen Tokens --> Automatisches herunterladen von "kaggle.json"
        - Die JSON Datei in einem Ordner "C:Users/.kaggle" speichern
        - Daten als CSV herunterladen
        - Speichern als der Funktion als .py Datei

    2. Einlesen, Bereinigen und Speichern der Daten
        - Die CSV-Datei einlesen
        - Entfernung von nicht relevanten/unnötigen Zeilen und Spalten
        - Löschen der Zeilen mit fehlenden Daten --> Alle Spalten besitzen die gleiche Länge
        - Dubletten entfernt
        - Zeilen mit unlogischen/fehlerhaften Daten gelöscht
        - Speichern als der Funktion als .py Datei

    3. Machine Learning
        - Implementierung und Training der Machine Learning Algorithmen für ein einziges Fahrzeugmodel (Volkswagen Golf)
        - Folgende relevante Machine Learning Methoden werden eingesetzt:
            - Lineare Regression
            - Polynomische Regression
            - Entscheidungsbaumregression
            - Random Forest Regression
            - Gradient Boosting Regression
            - Deep Learning
        - Als Features für die Modelle werden Kilometerstand, Leistung und Baujahr verwendet
        - Als Label wird der Preis gewählt
        - Nur Fahrzeugmodelle die mehr als 50 Einträge/Werte haben, werden berücksichtigt, alle anderen werden herausgefiltert
        - Finetuning der Hyperparameter (Grid Search)
        - Berechnung des Bestimmtheitsmaßes R^2
        - Ermittlung des besten KI-Verfahrens anhand von R^2
        - Plotten von tatsächlichen und prädizierten Daten

    4. GUI (Graphical User Interface)
        - Erstellung einer User Interface mittels QT
        - Auswahl der Marke und Modell
        - Eingabe von Daten wie Kilometerstand, Leistung und Baujahr
        - Ausgabe des geschätzten Preises anhand der besten ML-Methode (Random Forest Regression)

    Entwickler: Vasile Sirbu
    Datum: 28.03.2025
