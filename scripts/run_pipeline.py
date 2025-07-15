from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
# from src.train import train_model
# from src.evaluate import evaluate_model 

def main():
    print("="*60)
    print("Step 1: LOADING DATA")
    print("-"*60)
    data_path = load_data()
    
    print("="*60)
    print("Step 2: PREPROCESSING DATA")
    print("-"*60)
    train_data,test_data = preprocess_data(data_path)

    # print("="*60)
    # print("Step 3: Model Training")
    # model = train_model(X_train, y_train)

    # print("="*60)
    # print("Step 4: Evaluation")
    # evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
