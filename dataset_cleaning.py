def dataset_cleaning():
    """
    Dataset einlesen, bereinigen und speichern

    Input: csv-Datei mit Rohdaten
    Output: csv-Datei mit bereinigten Daten

    Version: 1.0
    Entwickler: Vasile Sirbu
    Datumm: 16.03.2025
    """

    import pandas as pd


    #--------------------------------------------------- Einlesen -------------------------------------------------------------------------

    # Dataset als DataFrame einlesen
    # Kopfzeile ersetzen
    dataset = pd.read_csv("autoscout24-germany-dataset.csv",
                        header = 0,
                        names = ["Kilometerstand", "Marke", "Model", "Kraftstoff", "Getriebe", "Typ", "Preis", "Leistung", "Baujahr"]
                        )


    #--------------------------------------------------- Bereinigen -------------------------------------------------------------------------

    # Unnötige Spalte und Zeilen löschen
    dataset = dataset.drop("Typ", axis=1)
    dataset = dataset.drop(dataset[dataset["Marke"] == "Others"].index, axis = 0)

    # Die ersten 10 Zeilen darstellen
    print(dataset.head(10))

    # Anzahl der Elemente für jede Spalte anzeigen lassen
    print(dataset.count())


    # Die Zeilen mit fehlnden/korrupten Daten löschen, so dass alle Spalten gleiche Anzahl an Elemente besitzen
    dataset = dataset.drop(dataset[dataset["Getriebe"].isnull()].index, axis=0)
    print(dataset.count())

    dataset = dataset.drop(dataset[dataset["Model"].isnull()].index, axis=0)
    print(dataset.count())

    dataset = dataset.drop(dataset[dataset["Leistung"].isnull()].index, axis=0)
    print(dataset.count())

    # Dubletten ermitteln und anzeigen
    print(dataset[dataset.duplicated(keep=False)])

    # Dubletten löschen
    dataset.drop_duplicates(keep="first", inplace=True)
    print(dataset.count())

    # Unlogische Daten löschen
    dataset = dataset.drop(dataset[dataset["Leistung"] == 1].index, axis = 0)


    #--------------------------------------------------- Speichern -------------------------------------------------------------------------

    # Bereinigtes Dataset in csv Format speichern
    dataset.to_csv("cleaned_autoscout24-germany-dataset.csv")

