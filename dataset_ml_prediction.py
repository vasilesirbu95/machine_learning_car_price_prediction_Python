def dataset_ml_prediction():
    """
    Vorhersagen mit Hilfe von Machine Learning Verfahren für ein einziges Fahrzeugmodel erstellen.
    Folgende ML-Methoden werden eingesetzt:
        - Lineare Regression
        - Nicht-lineare Regression
        - Deep Learning

    
    Input: csv-Datei mit bereinigten Daten
    Output: Vorhergesagte Preise

    Version: 1.0
    Entwickler: Vasile Sirbu
    Datumm: 16.03.2025
    """
        
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn.linear_model as lm
    from pathlib import Path
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer


    #--------------------------------------------------- Einlesen -------------------------------------------------------------------------
    dataset = pd.read_csv("cleaned_autoscout24-germany-dataset.csv")


    #--------------------------------------------------- Gruppieren -------------------------------------------------------------------------

    # Statistische Beschreibung der Daten 
    print(dataset.describe())

    # Die Daten für ein bestimmtes Fahrzeug gruppieren und sortieren
    data_audi_a4 = dataset[(dataset["Marke"] == "Audi") & (dataset["Model"] == "A4")]
    print(data_audi_a4)

    #--------------------------------------------------- Lineare Regression -------------------------------------------------------------------------
    # Zuweisen der Linearen Regression einer Variable
    model = LinearRegression()

    # Ordner zum Speichern der Ergebnisse
    save_path = Path("./ML_AUDI_A4")

    # Name des Plots
    name_plt = 'AUDI_A4.png'

    # Speicher-Pfad für die erstellte Abbildung
    path_plot = save_path.joinpath(name_plt)

    # Speichern von X in ein bestimmtes Format X = [[Element 1], [Element 2],...]
    #X = data_audi_a4["Baujahr"].to_numpy().reshape(-1,1)
    X = data_audi_a4[["Baujahr", "Kilometerstand", "Leistung", "Getriebe", "Kraftstoff"]]

    # Transformation der Spalten, um qualitative Features erfassbar zu machen
    cf = ColumnTransformer([("Getriebe_Kraftstoff", OneHotEncoder(drop = "first"), ["Getriebe", "Kraftstoff"])], remainder = "passthrough")

    # Fitten von cf
    cf.fit(X)

    # Transformation von X
    X_transformed = cf.transform(X)

    # Speichern des Outputs in die Variable y
    y = data_audi_a4["Preis"]

    # Aufteilen in Train und Test Daten
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, train_size=0.8, random_state=42)

    # Trainieren des Modells
    model.fit(X_train,y_train)

    # Bestimmtheitsmaß r^2
    print("Train: ", model.score(X_train,y_train))
    print("Test: ", model.score(X_test,y_test))

    # Prädizierte Werte für alle Input-Einträge berechnen
    predicted_price = model.predict(X_test) 

    # Plotten der tatsächlichen und prädizierten Werte in einem einzigen Plot
    plt.scatter(data_audi_a4['Baujahr'], data_audi_a4['Preis'], label=name_plt)
    plt.plot(X_test["Baujahr"], predicted_price, color = "red")
    plt.title(name_plt)
    plt.xlabel('Baujahr')
    plt.ylabel('Preis')
    plt.legend()
    plt.show()
    plt.savefig(path_plot)
    #plt.close()