from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

def evaluate_model(model, test_data,save_fig = True,save_report = True):
    y_test = test_data['Class']
    X_test = test_data.drop('Class', axis=1)

    threshold = 0.09090869
    # Predict using probability + threshold
    y_probs = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_probs >= threshold).astype(int)
    print("-> Prediction done...")

    # Metrics
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    
    print(f"-> Evaluation on the test set:= ")

    print(f"Threshold used      : {threshold:.4f}")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Precision           : {prec:.4f}")
    print(f"Recall              : {rec:.4f}")
    print(f"F1 Score            : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Save classification report and metrics to JSON
    if save_report:
        metrics_dict = {
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True)
        }

        json_path = "./outputs/reports/metrics_report.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        print(f"-> Metrics and classification report saved to {json_path}")


    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save and show
    if save_fig:
        os.makedirs('./outputs/figures', exist_ok=True)
        plt.savefig('./outputs/figures/confusion_matrix.png', bbox_inches='tight')
    plt.show()
