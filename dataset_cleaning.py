"""
Dataset einlesen, bereinigen und visualisieren
"""

# Entwickler: Vasile Sirbu
# Datumm: 12.03.2025

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset als DataFrame einlesen
# Kopfzeile ersetzen
dataset = pd.read_csv("autoscout24-germany-dataset.csv",
                      header = 0,
                      names = ["Kilometerstand", "Marke", "Model", "Kraftstoff", "Getriebe", "Typ", "Preis", "Leistung", "Baujahr"]
                      )

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

# Statistische Beschreibung der Daten 
print(dataset.describe())

print(dataset.groupby("Marke").describe())



#sns.histplot(dataset["Leistung"])
#plt.show()