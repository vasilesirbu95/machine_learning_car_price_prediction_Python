import kaggle

kaggle.api.authenticate()

#print(kaggle.api.dataset_list_files('asinow/car-price-dataset').files)

kaggle.api.dataset_download_files('asinow/car-price-dataset', path = '.', unzip = True)

kaggle.api.dataset_metadata('asinow/car-price-dataset', path = '.')

datasets = kaggle.api.dataset_list(search = "car-price")
print(datasets)