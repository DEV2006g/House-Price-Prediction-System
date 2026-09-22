# House Price Prediction System

## Skillairo AI Internship — Minor Project

**Project:** House Price Prediction System  
**Domain:** Artificial Intelligence / Machine Learning  
**Author:** Chetany Kalaneya

---

## 1. Project Overview

The House Price Prediction System is an end-to-end machine learning application that predicts residential house prices from property-related features.

The project uses the Kaggle **House Prices — Advanced Regression Techniques (Ames Housing)** dataset and covers:

- Data preprocessing
- Feature engineering
- Exploratory data analysis
- Regression model training
- Cross-validation
- Model comparison
- Error and residual analysis
- Feature importance analysis
- Interactive house valuation

The final system provides an interactive web interface for estimating the market value of a property.

---

## 2. Problem Statement

House prices depend on many factors such as living area, overall quality, number of rooms, bathrooms, garage capacity, neighborhood, construction year, and property condition.

The objective is to develop a regression model that learns these relationships from historical housing data and predicts the expected sale price of a new property.

---

## 3. Dataset

**Dataset:** House Prices — Advanced Regression Techniques

**Source:** Kaggle

Dataset page:  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The project uses the `SalePrice` column as the prediction target.

The prepared training dataset contains **1,458 properties** after project-specific preparation/outlier screening.

---

## 4. Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn

### Visualization
- Matplotlib

### Web Application
- Flask
- HTML
- CSS
- JavaScript

### Model Persistence
- Joblib

---

## 5. Machine Learning Models

The project evaluates multiple regression algorithms:

1. Ridge Regression
2. Lasso Regression
3. ElasticNet
4. Random Forest
5. Gradient Boosting
6. EnsembleBlend

Models are compared using cross-validation and regression evaluation metrics.

---

## 6. Feature Engineering

The project creates domain-relevant features such as:

- `TotalSF` — combined living/floor space
- `TotalBath` — combined bathroom measure
- `HouseAge` — property age
- `RemodelAge` — age since remodeling
- Property condition/amenity indicators

Categorical variables are encoded and numerical variables are processed through the machine learning pipeline.

---

## 7. Model Evaluation

The main evaluation metrics are:

- R² Score
- RMSLE
- RMSE
- MAE

### Final Model

**EnsembleBlend**

Reported evaluation:

- **R²:** 0.9355
- **RMSLE:** 0.1108
- **RMSE:** $20,176.91
- **MAE:** $13,122.09

These values are based on the supplied project implementation and evaluation artifacts.

---

## 8. Application Features

The HousePriceAI application provides:

- Property valuation calculator
- Quick property presets
- Interactive property inputs
- Estimated market value
- Prediction range
- Model information
- Key value drivers
- Model benchmarking
- Diagnostic plots
- Feature importance
- Kaggle submission functionality

---

## 9. Project Structure

```text
House Price Prediction System/
│
├── app/
├── data/
├── models/
├── notebooks/
├── static/
├── templates/
├── tests/
├── requirements.txt
├── app.py
└── README.md
```

The exact folders may vary slightly depending on the project package version.

---

## 10. How to Run

### Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd House-Price-Prediction-System
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

### Step 3 — Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the application

```bash
python app.py
```

Open the local URL shown by Flask in your browser.

---

## 11. Evaluation & Evidence

The project includes:

- Model benchmark results
- Actual vs. predicted price plot
- Residual analysis
- Feature importance visualization
- Interactive valuation interface
- Model performance dashboard

---

## 12. Learning Outcomes

Through this project, the following skills were demonstrated:

- Regression modeling
- Feature engineering
- Data preprocessing
- Cross-validation
- Model evaluation
- Error analysis
- Feature importance analysis
- Flask application development
- Machine learning model deployment

---

## 13. Internship Submission

**Internship:** Skillairo AI Internship  
**Project Type:** Minor Project  
**Submission Deadline:** 20 October 2026

**GitHub Repository:**  
`<ADD YOUR VERIFIED GITHUB LINK HERE>`

**Hosted Application:**  
`<ADD HOSTED LINK IF AVAILABLE>`

---

## 14. Author

**Chetany Kalaneya**

Artificial Intelligence / Machine Learning Enthusiast
