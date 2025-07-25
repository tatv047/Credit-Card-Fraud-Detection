"""downoading the data""" 
import kagglehub # type: ignore
import shutil
import os 
import pandas

def load_data():

    print("-> Downloading the data...")
    #download the data
    mlg_ulb_creditcardfraud_path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
    print('-> Data source import complete.')
    print(f"-> downloaded data-location: {mlg_ulb_creditcardfraud_path}")

    #create the folders if they don't exist
    os.makedirs("./data/raw",exist_ok= True)
    os.makedirs("./data/processed",exist_ok = True)
    print("->'data' folder present/created")

    src_path = './data/raw/creditcard.csv'
    if not os.path.exists(src_path):
        shutil.copy(
            os.path.join(mlg_ulb_creditcardfraud_path, 'creditcard.csv'),
            src_path
        )
        print('-> Data stored in working directory.')
    else:
        print("-> File already exists...")

    #store the data
    
    shutil.copy(
        os.path.join(mlg_ulb_creditcardfraud_path,'creditcard.csv'),
        src_path
    )
    
    return src_path





