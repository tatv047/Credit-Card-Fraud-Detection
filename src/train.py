from catboost import CatBoostClassifier # type: ignore
import os

def train_model(train_data):

    y_train = train_data['Class']
    X_train = train_data.drop('Class',axis = 1)

    neg = y_train.value_counts()[0]
    pos = y_train.value_counts()[1]
    pos_weight = neg/pos

    # parameters of the model
    params = {
                'iterations': 249,
                'depth': 9,
                'learning_rate': 0.10365129915403434,
                'l2_leaf_reg': 5.6721319452291885,
                'border_count': 170,
                'class_weights': [1, pos_weight],
                'random_state': 42,
                'verbose': 0
                # 'task_type': 'GPU',
                # 'devices': '0'
            }
    print("-> Model training started...")
    model = CatBoostClassifier(**params)
    model.fit(X_train,y_train)

    print("-> Model training done...")
    print("-> Parameters of the 'CatBoost' model..")
    print(params)

    model_path = './outputs/models/catboost_model.cbm'
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path,format="cbm")
        print(f"-> Model saved to {model_path}")

    return model 

