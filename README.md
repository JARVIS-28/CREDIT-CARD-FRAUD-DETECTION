# Credit Card Fraud Detection

## Objective

The objective of this project is to build a machine learning model that can distinguish fraudulent credit card transactions from legitimate ones.

## Dataset

The dataset used for this project can be found on Kaggle:
[https://www.kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

It contains transaction details such as amount, user ID, merchant information, timestamps, etc.

## Project Structure

```
credit-card-fraud-detection/
│
├── data/                    # Raw dataset (fraudTest.csv, fraudTrain.csv)
├── notebooks/               # Jupyter notebooks for EDA & modeling (EDA.ipynb)
├── src/                     # Python scripts for modular code (preprocessing.py, model.py)
├── results/                 # Plots, reports
├── README.md
├── requirements.txt
└── main.py
```

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd credit-card-fraud-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1.  **Exploratory Data Analysis (Optional):**
    Open and run the `notebooks/EDA.ipynb` notebook using Jupyter Lab, Jupyter Notebook, or VS Code.

2.  **Run the main training and evaluation pipeline:**
    Execute `main.py` from your terminal within the project's root directory.

    *   **Basic Run (uses default Logistic Regression, runs preprocessing):**
        ```bash
        python main.py
        ```

    *   **Specify Model Type (e.g., XGBoost):**
        ```bash
        python main.py --model_type xgboost
        ```
        Available models: `logistic_regression`, `random_forest`, `xgboost`, `gradient_boosting`.

    *   **Use Pre-processed Data (if available):**
        Add the `--use_processed_data` flag to skip the preprocessing step and load from `data/*_processed.csv`.
        ```bash
        python main.py --model_type random_forest --use_processed_data
        ```

    *   **Generate SHAP Explanations:**
        Add the `--explain` flag to generate SHAP summary and force plots after evaluation.
        ```bash
        python main.py --model_type xgboost --explain --use_processed_data
        ```

3.  **Check Results:**
    Find evaluation metrics (`.txt`), plots (`.png`), and the saved model (`.joblib`) in the `results/` directory, named according to the model type used.

## Expected Outcome

A fraud detection system that minimizes false positives while maximizing fraud detection accuracy. The project will also include an analysis of misclassifications using explainability techniques.

## Metrics Summary

*(To be filled in after model evaluation)*

## Sample Outputs/Plots

*(To be added after running the analysis and model training)*
