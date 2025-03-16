def api_download_dataset_kaggle():

    """Beschreibung: Diese Datei ladet den Dataset von Kaggle mit Hilfe eines API Tokens herunter
    
    Version: 1.0
    Entwickler: Vasile Sirbu
    Datum: 16.03.2025
    
    """



    #--------------------------------------------------------------------------------------------------------------------
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

    dataset_url = "kalyankumarm/germany-cars-dataset"

    # Herunterladen der Daten (germany_auto_industry_kaggle.csv)
    kaggle.api.dataset_download_files(dataset_url, path = '.', unzip = True)

    # Herunterladen der Metadaten (dataset-metadata.json)
    pt = kaggle.api.dataset_metadata(dataset_url, path = '.')
   
    #--------------------------------------------------------------------------------------------------------------------