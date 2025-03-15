"""
Dataset einlesen, bereinigen und visualisieren
"""

# Entwickler: Vasile Sirbu
# Datumm: 12.03.2025

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from pathlib import Path



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

print("GROUP BY\n",dataset.groupby("Marke")["Preis"])

data_jag_xf = dataset[(dataset["Marke"] == "Jaguar") & (dataset["Model"] == "XF")]

print(data_jag_xf)

#------------------------------------------------------------------------------------------------------
# Lineare Regression

grouped = dataset.groupby(["Marke", "Model"])
save_path = Path('C:/Users/admin/git_project/Bilder/')

lr = lm.LinearRegression()

# Ermittlung der linearen Regression für jedes Model
# Das Speichern der linearen Kennlinie als PNG-Bild in dem Ordner "Bilder"
for name, group in grouped:
    name_plt = name[0]+"_"+name[1]
    if "/" in name_plt:
        name_plt = name_plt.replace("/", "_")
    path_plot = save_path.joinpath(name_plt+'.png')
    X = []
    for x in group["Baujahr"]:
        X.append([x])
    y = group["Preis"]
    lr.fit(X,y)
    lr.score(X,y)
    predicted_price = lr.predict(X) 
    plt.scatter(group['Baujahr'], group['Preis'], label=name_plt)
    plt.plot(X, predicted_price, color = "red")
    plt.title(name_plt)
    plt.xlabel('Baujahr')
    plt.ylabel('Preis')
    plt.legend()
    plt.savefig(path_plot)
    plt.close()

    

#------------------------------------------------------------------------------------------------------
# Plotten / Visualisierung

#plt.scatter(X,y)
#plt.plot(X, predicted_price, color = "red")
#plt.title(data_jag_xf["Marke"].unique()[0]+" "+data_jag_xf["Model"].unique()[0])
#plt.xlabel("Baujahr")
#plt.ylabel("Preis in €")
#plt.show()

