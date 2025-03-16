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

    #--------------------------------------------------- Einlesen -------------------------------------------------------------------------
    dataset = pd.read_csv("cleaned_autoscout24-germany-dataset.csv")


    #--------------------------------------------------- Gruppieren -------------------------------------------------------------------------

    # Statistische Beschreibung der Daten 
    print(dataset.describe())

    # Die Daten für ein bestimmtes Fahrzeug gruppieren und sortieren
    data_audi_a4 = dataset[(dataset["Marke"] == "Audi") & (dataset["Model"] == "A4")]

    #--------------------------------------------------- Lineare Regression -------------------------------------------------------------------------
    # Zuweisen der Linearen Regression einer Variable
    lr = LinearRegression()

    # Ordner zum Speichern der Ergebnisse
    save_path = Path("./ML_AUDI_A4")

    # Name des Plots
    name_plt = 'AUDI_A4.png'

    # Speicher-Pfad für die erstellte Abbildung
    path_plot = save_path.joinpath(name_plt)

    # Speichern von X in ein bestimmtes Format X = [[Element 1], [Element 2],...]
    X = []
    for x in data_audi_a4["Baujahr"]:
        X.append([x])

    # Speichern des Outputs in die Variable y
    y = data_audi_a4["Preis"]

    # Trainieren des Modells
    lr.fit(X,y)

    print(lr.score(X,y))

    # Prädizierte Werte für alle Input-Einträge berechnen
    predicted_price = lr.predict(X) 

    # Plotten der tatsächlichen und prädizierten Werte in einem einzigen Plot
    plt.scatter(data_audi_a4['Baujahr'], data_audi_a4['Preis'], label=name_plt)
    plt.plot(X, predicted_price, color = "red")
    plt.title(name_plt)
    plt.xlabel('Baujahr')
    plt.ylabel('Preis')
    plt.legend()
    plt.show()
    plt.savefig(path_plot)
    #plt.close()



