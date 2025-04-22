import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_data(test_path):
    df = pd.read_csv(test_path)
    X = df.drop(columns=['is_fraud'], errors='ignore')  # keep cc_num

    y = df['is_fraud']
    return X, y

def run_inference(model_path, test_path):
    print("Loading model...")
    model = joblib.load(model_path)

    print("Loading test data...")
    X_test, y_test = load_data(test_path)

    print("Running predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    model_path = "E:/growthLinks/CREDIT-CARD-FRAUD-DETECTION/results/xgb_model.pkl"  # your saved model path
    test_path = "E:/growthLinks/CREDIT-CARD-FRAUD-DETECTION/data/processed_fraudTest.csv"  # update if different

    run_inference(model_path, test_path)
