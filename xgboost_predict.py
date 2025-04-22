import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_data(test_path):
    """
    Loads and returns test features and labels from a CSV file.

    Parameters:
    test_path (str): Path to the test CSV file.

    Returns:
    X (DataFrame): Features for testing.
    y (Series): True labels.
    """
    df = pd.read_csv(test_path)
    X = df.drop(columns=['is_fraud'], errors='ignore')  # keep cc_num
    y = df['is_fraud']
    return X, y


def run_inference(model_path, test_path):
    """
    Loads the model and test data, runs predictions, evaluates the model,
    and generates SHAP summary plot for explainability.

    Parameters:
    model_path (str): Path to the saved model (.pkl file).
    test_path (str): Path to the processed test CSV file.
    """
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

    # Model explainability using SHAP
    print("Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap_summary_plot.png")
    print("SHAP summary plot saved as 'shap_summary_plot.png'.")


if __name__ == "__main__":
    # Update these paths as per your directory structure
    model_path = "E:/growthLinks/CREDIT-CARD-FRAUD-DETECTION/results/xgb_model.pkl"
    test_path = "E:/growthLinks/CREDIT-CARD-FRAUD-DETECTION/data/processed_fraudTest.csv"

    run_inference(model_path, test_path)
    print("Inference completed.")
    print("Model evaluation and SHAP summary plot generation completed.")
    print("Results saved in the current directory.")
    print("Exiting...")