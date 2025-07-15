"""downoading the data""" 
import kagglehub # type: ignore
import shutil
import pandas as pd 
import os 
import warnings
warnings.filterwarnings("ignore")

print('='*60)
print("Downloading the data")
print('-'*60)

#create the folders if they don't exist
os.makedirs("./data/raw",exist_ok= True)
os.makedirs("./data/processed",exist_ok = True)
print("->'data' folder present/created")

#download the data
mlg_ulb_creditcardfraud_path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
print('-> Data source import complete.')
print(f"-> downloaded data-location: {mlg_ulb_creditcardfraud_path}")

#store the data
src_path = './data/raw/creditcard.csv'
shutil.copy(
    os.path.join(mlg_ulb_creditcardfraud_path,'creditcard.csv'),
    src_path
)
print('-> Data stored in working directory.')



"""Preprocessing the data"""
print('='*60)
print("Preprocessing the data")
print('-'*60)

import pandas as pd 

# load the data
df = pd.read_csv('./data/raw/creditcard.csv')

# handling nulls
if df.isnull().sum().sum() != 0:
    df.dropna(axis=0,inplace=True)

# handling duplicates
if df.duplicated().sum() != 0:
    df.drop_duplicates(inplace=True)

df.to_csv('./data/processed/cleaned_df.csv',index=False)
print("-> Cleaned the raw data and saved it.")


