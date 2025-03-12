"""
Dataset einlesen und bereinigen 
"""

# Entwickler: Vasile Sirbu
# Datumm: 12.03.2025

import pandas as pd
import seaborn as sns

dataset = pd.read_csv("autoscout24-germany-dataset.csv")

print(dataset.head())

sns.histplot(dataset)