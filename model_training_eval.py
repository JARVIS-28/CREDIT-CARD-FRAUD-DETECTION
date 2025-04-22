import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
import joblib
import shap
import matplotlib.pyplot as plt
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def save_model_report_pdf(model_name, classification_rep, conf_matrix, pr_auc, roc_auc, shap_plot_path, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Model Evaluation Report: {model_name}", styles['Title']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Classification Report:", styles['Heading2']))
    story.append(Paragraph(f"<pre>{classification_rep}</pre>", styles['Code']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Confusion Matrix:", styles['Heading2']))
    story.append(Paragraph(str(conf_matrix), styles['Code']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"PR AUC: {pr_auc:.4f}", styles['Normal']))
    story.append(Paragraph(f"ROC AUC: {roc_auc:.4f}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("SHAP Summary Plot:", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Image(shap_plot_path, width=6*inch, height=4*inch))
    doc.build(story)

# Load processed data
train_path = "data/fraudTrain_processed.csv"
test_path = "data/fraudTest_processed.csv"
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

X_train = df_train.drop("is_fraud", axis=1)
y_train = df_train["is_fraud"]
X_test = df_test.drop("is_fraud", axis=1)
y_test = df_test["is_fraud"]

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Define models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "xgboost": XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), use_label_encoder=False, eval_metric="logloss", random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    class_report = classification_report(y_test, y_pred, digits=4)
    conf_matrix = confusion_matrix(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else 0.0
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

    print(f"\nClassification Report for {name}:")
    print(class_report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("PR AUC:", pr_auc)
    print("ROC AUC:", roc_auc)

    # Save the model
    joblib.dump(model, f"results/{name}_model.joblib")

    # SHAP explanation (global feature importance)
    print(f"Generating SHAP summary for {name}...")
    explainer = shap.Explainer(model, X_train_res)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    shap_plot_path = f"results/{name}_shap_summary.png"
    plt.savefig(shap_plot_path)
    plt.close()

    # Save PDF report
    pdf_path = f"results/{name}_report.pdf"
    save_model_report_pdf(
        model_name=name,
        classification_rep=class_report,
        conf_matrix=conf_matrix,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        shap_plot_path=shap_plot_path,
        output_path=pdf_path
    )

print("\nAll models trained, evaluated, explained, and reports saved as PDFs.")
