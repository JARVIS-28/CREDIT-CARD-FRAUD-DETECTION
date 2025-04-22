# Credit Card Fraud Detection

## Overview
This project builds a machine learning pipeline to detect fraudulent credit card transactions using advanced feature engineering, class imbalance handling, and model explainability.

## Project Structure
```
CREDIT-CARD-FRAUD-DETECTION/
│
├── data/                    # Raw and processed datasets (fraudTrain.csv, fraudTest.csv, processed_fraudTrain.csv, processed_fraudTest.csv)
├── notebooks/               # Jupyter notebooks for EDA, preprocessing, and model training (growthLink.ipynb, EDA.ipynb)
├── results/                 # Model artifacts (e.g., xgb_model.pkl, rf_model.pkl, logreg_model.pkl)
├── requirements.txt         # Python dependencies
├── xgboost_predict.py       # Script for making predictions using the XGBoost model
├── README.md
```

## Workflow

1. **Exploratory Data Analysis:**
   - Use `notebooks/EDA.ipynb` to:
     - Analyze the data (EDA)
     - Visualize distributions and correlations
     - Understand class imbalance and feature relationships

2. **Preprocessing, Model Training, and Evaluation:**
   - Use `notebooks/growthLink.ipynb` to:
     - Preprocess and engineer features
     - Handle class imbalance
     - Train and evaluate multiple models (Logistic Regression, Random Forest, XGBoost, etc.)
     - Save processed datasets (`processed_fraudTrain.csv`, `processed_fraudTest.csv`) and trained model files (e.g., `xgb_model.pkl`)
     - Summarize which model to use based on evaluation metrics

3. **Prediction:**
   - Use `xgboost_predict.py` to:
     - Load a trained XGBoost model and processed test data
     - Run predictions and print evaluation metrics (classification report, confusion matrix, ROC-AUC)

   - Example:
     ```bash
     python xgboost_predict.py
     ```
     
## Key Features
- **Separation of EDA and Modeling:**
  - EDA in `EDA.ipynb`, full pipeline in `growthLink.ipynb`
- **Processed Data and Models:**
  - Outputs processed CSVs and model `.pkl` files for easy reuse
- **Prediction Script:**
  - Simple script for running predictions and evaluating the XGBoost model
  **Model Explainability:**
  - SHAP values are used to interpret XGBoost model predictions.
  - Visualizations show which features contribute to fraud detection and misclassifications.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage Example
```bash
# Run EDA
# (Open and run notebooks/EDA.ipynb in Jupyter or Colab)

# Run preprocessing, training, and evaluation
# (Open and run notebooks/growthLink.ipynb in Jupyter or Colab)

# Run prediction script for XGBoost
python xgboost_predict.py
```

## Notes
- The target column is `is_fraud` (1: fraud, 0: non-fraud).
- The pipeline is designed for extensibility and can be adapted for new features or models.
- You can adapt `xgboost_predict.py` for other models by changing the model path and file names.
