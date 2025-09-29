# 🍷 MLOps with MLflow - Random Forest Experiments

This repository demonstrates how to use **MLflow** for end-to-end **MLOps experiment tracking**.  
It includes **Random Forest experiments** on the **Wine dataset** and **Breast Cancer dataset**, covering hyperparameter tuning, metrics logging, artifact management, and model tracking.

---

## 📂 Project Structure

```
MLOps-MLflow-RandomForest/
│── src/ # Source code
│ ├── breast_cancer_rf.py # Random Forest + GridSearchCV on Breast Cancer dataset
│ ├── wine_classification.py # Random Forest experiment on Wine dataset
│ ├── test.py # Additional experiment logging
│
│── mlruns/ # MLflow runs directory (auto-generated)
│── mlartifacts/ # MLflow artifacts (models, plots, etc.)
│── Confusion_matrix.png # Saved confusion matrix example
│── .gitignore # Git ignore file
│── README.md # Project documentation
```


---

## ⚡ Features
- ✅ **MLflow Experiment Tracking** – Log parameters, metrics, artifacts, and models  
- ✅ **GridSearchCV** for hyperparameter tuning  
- ✅ **Random Forest Classifier** on multiple datasets  
- ✅ **Confusion Matrix Visualization** with Seaborn & Matplotlib  
- ✅ **Dagshub Integration** for remote experiment tracking  
- ✅ **MLOps-ready pipeline** (logging, tracking, reproducibility)

---



### 🔧 Prerequisites
- Python 3.8+
- Install dependencies:

pip install -r requirements.txt

## ▶️ Run Experiments

# Breast Cancer Experiment
python src/breast_cancer_rf.py

# Wine Classification Experiment
python src/wine_classification.py

## 📊 View MLflow UI

mlflow ui

## 🔗 DagsHub Integration

- This repo is integrated with DagsHub for experiment tracking:
- Remote MLflow tracking URI
- Centralized logging of metrics & artifacts

## 👨‍💻 Author

Shayan
AI & ML Specialist | MLOps Enthusiast
