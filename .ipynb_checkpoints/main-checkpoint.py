from api_download_dataset_kaggle import api_download_dataset_kaggle
from dataset_cleaning import dataset_cleaning
from dataset_ml_prediction import dataset_ml_prediction

path_data_csv = "cleaned_autoscout24-germany-dataset.csv"

#--------------------------------------------------- Herunterladen der Daten -------------------------------------------------------------------------
api_download_dataset_kaggle()


#--------------------------------------------------- Bereinigen der Daten -------------------------------------------------------------------------
dataset_cleaning()


#--------------------------------------------------- Linear Regression -------------------------------------------------------------------------
linear_regression_model(path_data_csv)


#--------------------------------------------------- Deep Learning -------------------------------------------------------------------------
deep_learning_model(path_data_csv)