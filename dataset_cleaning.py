"""
Dataset einlesen, bereinigen und visualisieren
"""

# Entwickler: Vasile Sirbu
# Datumm: 12.03.2025

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm



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

data_jag_xf = dataset[(dataset["Marke"] == "Jaguar") & (dataset["Model"] == "XF")]

print(data_jag_xf)

#------------------------------------------------------------------------------------------------------
# Lineare Regression
X = []
for x in data_jag_xf["Baujahr"]:
    X.append([x])

y = data_jag_xf["Preis"]
lr = lm.LinearRegression()
lr.fit(X,y)
lr.score(X,y)
predicted_price = lr.predict(X)

#------------------------------------------------------------------------------------------------------
# Plotten / Visualisierung

plt.scatter(X,y)
plt.plot(X, predicted_price, color = "red")
plt.title(data_jag_xf["Marke"].unique()[0]+" "+data_jag_xf["Model"].unique()[0])
plt.xlabel("Baujahr")
plt.ylabel("Preis in €")
plt.show()

