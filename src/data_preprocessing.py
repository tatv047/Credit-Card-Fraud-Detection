import pandas as pd 
import os

def preprocess_data(data_path):

    # load the data
    df = pd.read_csv(data_path)
    if os.path.exists("./data/processed/train.csv") and os.path.exists("./data/processed/test.csv"):
        print("-> Train/test files already exist. Skipping preprocessing...")
        train_data = pd.read_csv("./data/processed/train.csv")
        test_data = pd.read_csv("./data/processed/test.csv")
        return train_data, test_data

    print("-> raw data loaded... ")


    print("Data Cleaning :=")
    # handling nulls
    if df.isnull().sum().sum() != 0:
        df.dropna(axis=0,inplace=True)
    print("-> Nulls handled...")

    # handling duplicates
    if df.duplicated().sum() != 0:
        df.drop_duplicates(inplace=True)
    print("-> Duplicates handled...")


    print("Feature Engineering:= ")
    # converting time
    df['Time'] = pd.to_datetime(df['Time'],unit = 's')

    # creating new features
    df['day'] = df['Time'].dt.day
    df['hour'] = df['Time'].dt.hour
    df['min'] = df['Time'].dt.minute
    print("-> Created three new features: day,hour,min from 'Time'...")

    # dropping the Time column
    df.drop(columns=['Time'],axis=1,inplace=True)

    df.to_csv('./data/processed/cleaned_df.csv',index=False)
    print("-> Preprocessed the raw data and saved it...")

    print("Training/testing dataset creation:= ")

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Creating the input feature matrix and target vector
    X = df.drop(columns=['Class']).copy()
    y = df['Class']
    print("-> Feature matrix and target vector created...")
    print(f"-> Feature matrix(X) shape: {X.shape}")
    print(f"-> Target vector(y) shape: {y.shape}")

    # Splitting the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify = y)
    print("-> Split the data into 80:20 train:test ratio...")
    print(f"-> training set: {X_train.shape}")
    print(f"-> test set: {X_test.shape}")
    print(f"-> Fraud rate in train: {y_train.mean():.4f}")
    print(f"-> Fraud rate in test: {y_test.mean():.4f}")

    # Scaling and converting back to DataFrame
    scaler = StandardScaler()

    X_train_scaled_array = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled_array,
                                columns=X_train.columns,
                                index=X_train.index)

    X_test_scaled_array = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled_array,
                                columns=X_test.columns,
                                index=X_test.index)

    X_train = X_train_scaled
    X_test = X_test_scaled
    print("-> Scaled X_train and X_test using StandardScaler...")

    # creaing the training and testing data
    train_data = X_train
    test_data = X_test
    train_data["Class"] = y_train
    test_data["Class"] = y_test

    # Save to CSV files
    train_data.reset_index(drop=True).to_csv("./data/processed/train.csv", index=False)
    test_data.reset_index(drop=True).to_csv("./data/processed/test.csv", index=False)
    print("-> Saved train.csv and test.csv under ./data/processed/ ")

    return train_data,test_data

