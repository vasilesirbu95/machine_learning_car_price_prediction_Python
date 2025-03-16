from api_download_dataset_kaggle import api_download_dataset_kaggle
from dataset_cleaning import dataset_cleaning
from dataset_ml_prediction import dataset_ml_prediction

#--------------------------------------------------- Herunterladen der Daten -------------------------------------------------------------------------
api_download_dataset_kaggle()


#--------------------------------------------------- Bereinigen der Daten -------------------------------------------------------------------------
dataset_cleaning()


#--------------------------------------------------- Machine Learning -------------------------------------------------------------------------
dataset_ml_prediction()