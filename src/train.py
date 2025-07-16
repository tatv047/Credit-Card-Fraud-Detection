from xgboost import XGBClassifier # type: ignore
import os

def train_model(train_data):
    model_path = './outputs/models/xgb_model.model'  # XGBoost uses .json or .model format

    if os.path.exists(model_path):
        print(f"-> Model already exists at {model_path}. Loading it instead of retraining...")
        model = XGBClassifier()
        model.load_model(model_path)
        return model

    # If no model exists
    y_train = train_data['Class']
    X_train = train_data.drop('Class', axis=1)

    neg = y_train.value_counts()[0]
    pos = y_train.value_counts()[1]
    pos_weight = neg / pos

    # parameters of the model (change this block as needed)
    params = {
        'n_estimators': 225,
        'max_depth': 8,
        'learning_rate': 0.2540831362105696,
        'subsample': 0.7114496210011669,
        'colsample_bytree': 0.7677641543248914,
        'min_child_weight': 3,
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
        # 'tree_method': 'hist',
        # 'device': 'cuda'
    }

    print("-> Model training started...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("-> Model training done...")
    print("-> Parameters of the 'XGBoost' model..")
    print(params)

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        print(f"-> Model saved to {model_path}")

    return model
