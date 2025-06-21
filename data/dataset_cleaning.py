def dataset_cleaning(raw_dataset_path):
    """
    Dataset einlesen, bereinigen und speichern

    Input: Pfad der Rohdaten
    Output: Pfad des bereinigten Datensets

    Version: 1.0
    Entwickler: Vasile Sirbu
    Datumm: 16.03.2025
    """

    import pandas as pd


    #--------------------------------------------------- Einlesen -------------------------------------------------------------------------

    # Dataset als DataFrame einlesen
    # Kopfzeile ersetzen
    dataset = pd.read_csv(raw_dataset_path,
                        header = 0,
                        names = ["Kilometerstand", "Marke", "Model", "Kraftstoff", "Getriebe", "Typ", "Preis", "Leistung", "Baujahr"]
                        )


    #--------------------------------------------------- Bereinigen -------------------------------------------------------------------------

    # Unnötige Spalte und Zeilen löschen
    dataset = dataset.drop("Typ", axis=1)
    dataset = dataset.drop(dataset[dataset["Marke"] == "Others"].index, axis = 0)

    # Die ersten 10 Zeilen darstellen
    # print(dataset.head(10))

    # Anzahl der Elemente für jede Spalte anzeigen lassen
    # print(dataset.count())


    # Die Zeilen mit fehlenden Daten löschen, so dass alle Spalten gleiche Anzahl an Elemente besitzen
    dataset.dropna(inplace=True)
    # print(dataset.count())

    # Dubletten ermitteln und anzeigen
    print(dataset[dataset.duplicated(keep=False)])

    # Dubletten löschen
    dataset.drop_duplicates(keep="first", inplace=True)
    # print(dataset.count())

    # Unlogische Daten löschen
    dataset = dataset.drop(dataset[dataset["Leistung"] == 1].index, axis = 0)


    #--------------------------------------------------- Speichern -------------------------------------------------------------------------t
    # Dateiname der bereinigten Datei erweitern
    cleaned_dataset_path = raw_dataset_path.split(".")[0] + '_cleaned.' + raw_dataset_path.split(".")[1]

    # Bereinigtes Dataset in csv Format speichern
    dataset.to_csv(cleaned_dataset_path)

    return cleaned_dataset_path

