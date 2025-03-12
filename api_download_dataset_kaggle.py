# Installation der kaggle-Bibliothek--> pip install kaggle

# Import
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Einlogen in Kaggle
# Profil --> Settings --> Create new token --> automatisches Download von "kaggle.json"
# .kaggle Ordner in "C:\Users\" oder "C:\Users\admin" erstellen
# "kaggle.json" in .kaggle einfügen

# Authetifizierung mit dem heruntergeladenen token
api = KaggleApi()
api.authenticate()

# Herunterladen der Daten (germany_auto_industry_kaggle.csv)
kaggle.api.dataset_download_files('heidarmirhajisadati/german-vehicle-price-and-efficiency-dataset', path = '.', unzip = True)

# Herunterladen der Metadaten (dataset-metadata.json)
kaggle.api.dataset_metadata('heidarmirhajisadati/german-vehicle-price-and-efficiency-dataset', path = '.')