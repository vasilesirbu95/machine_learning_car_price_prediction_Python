def ml_all():
    
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from pathlib import Path

#--------------------------------------------------- Einlesen -------------------------------------------------------------------------
dataset = pd.read_csv("cleaned_autoscout24-germany-dataset.csv")


#--------------------------------------------------- Sortieren -------------------------------------------------------------------------

# Statistische Beschreibung der Daten 
print(dataset.describe())

print("GROUP BY\n",dataset.groupby("Marke")["Preis"])

# Die Daten für ein bestimmtes Fahrzeug gruppieren und sortieren
data_audi_a4 = dataset[(dataset["Marke"] == "Audi") & (dataset["Model"] == "A4")]

plt.scatter(data_audi_a4["Baujahr"], data_audi_a4["Preis"])
plt.show()

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

