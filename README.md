# 🚖 UrbanGo Data Preprocessing for AI-Ready Data

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

## 📌 Project Overview
This repository contains the data preprocessing pipeline for **UrbanGo Assignment 1**, part of the **DSE ZG505 - Preprocessing for AI-Ready Data** coursework.

The objective was to take a raw, messy dataset simulating real-world ride-hailing data and apply systematic data cleaning, imputation, and transformation techniques to prepare it for machine learning models (specifically churn prediction).

## 🚀 Key Improvements & Results
The effectiveness of the preprocessing pipeline was measured using a churn prediction model baseline. The results show a significant improvement in model accuracy:

| Dataset State | Model Accuracy |
| :--- | :--- |
| **RAW Data** | 79.66% |
| **PREPROCESSED Data** | **83.50%** |

*Execution Status: VERIFIED & SUCCESS* ✅

## 🛠️ Data Preprocessing Pipeline

### 1. Data Quality Assessment (YData Profiling)
Generated a comprehensive HTML profiling report (`profiling_report.html`) to identify missing values, extreme skewness, kurtosis in numerical features (like `fare_amount` and `lifetime_rides`), and inconsistencies in categorical columns (e.g., casing issues in `payment_type`).

### 2. Handling Missing Data
*   **Customer Rating:** Created a missingness indicator column (`customer_rating_missing`) to preserve the pattern of missing data, followed by median imputation.
*   **Fare Amount:** Applied robust median imputation.
*   **Customer Age (KNN Imputation):** Implemented a scaled **3-Nearest-Neighbour (KNN)** algorithm using Euclidean distance across all other numeric features to accurately impute biologically missing or impossible age values without introducing data leakage from the target variable.

### 3. Outlier Handling & Transformation
*   **Impossible Records Removal:** Permanently dropped rows with biologically or physically impossible operational data (e.g., negative fares, distances > 200 km).
*   **Fare Capping:** Extreme fare surges were capped using the upper boundary of the **Interquartile Range (Q3 + 1.5 * IQR)**.
*   **Log Transformation:** Addressed the heavy right-tail distribution of `lifetime_rides` by applying a `log1p` transformation, compressing large positive values while safely retaining zero values, significantly reducing skewness.

### 4. Feature Engineering
*   **Contextual Baseline:** Computed the mean `distance_km` grouped by `city` to establish a contextual baseline metric.

## 📂 Repository Structure
*   `Assignment_A1_solution.py`: The core Python script containing the entire end-to-end preprocessing pipeline.
*   `UrbanGo_A1_cleaned.csv`: The final, AI-ready dataset output by the pipeline.
*   `profiling_report.html`: The extensive YData data profiling report highlighting the initial data quality issues.
*   `Part3_scorecard.txt`: The verified model performance scorecard.
*   `Assignment1_Report.docx`: Detailed project report.

## 🧑‍💻 Author
**Ajay Prakash**
* **Education:** M.Tech in Data Science and Engineering, BITS Pilani
* **Portfolio:** [ajayprakash.dev](https://ajayprakash.dev)
* **Student ID:** `2026nd04174@wilp.bits-pilani.ac.in`
* **License:** This project is licensed under the MIT License.
