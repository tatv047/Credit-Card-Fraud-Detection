# downoading and loading the data
import kagglehub # type: ignore
import shutil
import pandas as pd 
import os 

#create the folders if they don't exist
os.makedirs("./data/raw",exist_ok= True)
os.makedirs("./data/processed",exist_ok = True)
print('data folder present')

#download the data
mlg_ulb_creditcardfraud_path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
print('Data source import complete.')
print(f"downloaded data-location: {mlg_ulb_creditcardfraud_path}")

#store the data
src_path = './data/raw/creditcard.csv'
shutil.copy(
    os.path.join(mlg_ulb_creditcardfraud_path,'creditcard.csv'),
    src_path
)
print('Data stored in working directory.')