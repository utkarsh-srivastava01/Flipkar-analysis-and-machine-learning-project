# 💬 Customer Support CSAT Prediction and Analysis

A comprehensive data analysis and predictive modeling project focused on understanding the drivers of **Customer Satisfaction (CSAT)** in a customer support environment. This project covers deep exploratory data analysis (EDA) on support interactions and develops a Machine Learning model to predict CSAT scores based on support metrics.

---

## ✨ Project Overview

This repository contains the complete workflow for a customer support analytics project. The analysis aims to uncover the root causes of customer dissatisfaction and build a predictive tool for proactive intervention.

* **Goal:** Analyze support interaction data to uncover patterns and factors influencing Customer Satisfaction (CSAT) scores.
* **Prediction:** Build and evaluate a supervised learning model (specifically **XGBoost**) to predict the final CSAT score of a support interaction (on a 1-5 scale).
* **Objective:** Use the predictive model to flag high-risk interactions and help identify agent behaviors that lead to low satisfaction.

---

## 📊 Key Findings & Business Insights

The analysis was performed on **85,885 records** of customer support interactions.

| Key Metric | Insight | Source |
| :--- | :--- | :--- |
| **CSAT Distribution** | The dataset is highly imbalanced, with **69%** of interactions receiving the maximum score of **5** (high satisfaction). | EDA |
| **Baseline Accuracy** | A naive model guessing only the majority class (`CSAT=5`) would achieve **~69% Accuracy**, emphasizing the need for F1-Score. | EDA |
| **Feature Engineering** | Engineered features related to **response time** (`Issue_responded - Issue_reported at`) and **temporal factors** proved to be the most influential in predicting lower CSAT. | Model Evaluation |
| **Prediction Focus** | Model performance is focused on accurately classifying low CSAT scores (1, 2, 3), where intervention is critical. | Notebook |

---

## 🤖 Machine Learning Model Performance

An **XGBoost Classifier** was chosen for this project due to its robust performance on structured, imbalanced data. The problem was treated as a multi-class classification task (predicting CSAT scores 1 through 5).

| Metric | Model Performance | Interpretation |
| :--- | :--- | :--- |
| **Baseline Accuracy** | **0.69** | The model must significantly outperform this score (predicting only CSAT = 5) to be considered useful. |
| **Model Accuracy** | **[Insert Final Accuracy from Notebook]** | The overall percentage of correctly predicted CSAT scores. |
| **Macro F1-Score** | **[Insert Final Macro F1-Score from Notebook]** | The average F1-Score across all 5 CSAT classes, providing a reliable performance measure for imbalanced data. |

---

## 📁 Repository Structure

| File Name | Description |
| :--- | :--- |
| `ML projects.ipynb` | The main **Jupyter Notebook** containing all code for data cleaning, EDA, feature engineering, and the **Machine Learning** model training and evaluation. |
| `Customer_support_data.csv` | The raw dataset used for the entire analysis, containing customer interaction details and final CSAT scores. |
| `Xgboost_model.joblib` | The trained and serialized **XGBoost** model object (joblib format) for direct deployment. |
| `Xgboost_model.pkl` | The trained and serialized **XGBoost** model object (pickle format) for portability. |

---

## 🛠️ Technologies and Prerequisites

This project was built using the following tools and Python libraries:

* **Platform:** Jupyter Notebook
* **Language:** Python
* **Key Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `XGBoost`.

### Installation

To run the notebook locally and use the model, install the required libraries via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
