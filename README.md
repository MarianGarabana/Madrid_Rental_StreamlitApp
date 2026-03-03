# Madrid Rental Market — Interactive ML Dashboard

> A Streamlit application analyzing Madrid's rental housing market through machine learning.
> Built for the Machine Learning I course · MBDS · IE University · 2026

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Live App:** [madridrentalapp-mariangarabana.streamlit.app](https://madridrentalapp-mariangarabana.streamlit.app/)

---

## Overview

This dashboard brings a full end-to-end machine learning pipeline to life using ~2,100 rental listings sourced from Idealista (Madrid). It covers everything from exploratory data analysis to unsupervised clustering, rent prediction, and binary classification — all interactive, all trained live from the raw data at startup.

---

## Features

| Page | Description |
|---|---|
| **Market Explorer** | Filter listings by district and rent range. View rent distributions, median rent by district, rent vs. size scatter plots, and a correlation heatmap. |
| **Property Segments** | Explore 5 K-Means clusters with radar charts and segment profiles. Enter a property's attributes to classify it into its market segment. |
| **Rent Predictor** | OLS linear regression with VIF filtering and RFECV feature selection. Predict monthly rent with a 95% prediction interval and see where it sits in the market distribution. |
| **High Rent Classifier** | Logistic regression with interactive ROC curve, confusion matrix, odds ratios, and a probability gauge. Classify any property as High Rent (≥ €1,800). |

---

## ML Pipeline

### Phase 0 — EDA & Cleaning
- 2,100 listings → 2,089 after cleaning
- Fixed floor outlier (data entry error > 100)
- Engineered features: `Is_Special`, `Price_per_sqm`, `SqMt_per_Bed`, `District_Premium`, `Zone`, `Is_Central`, `Is_Studio`, `High_Rent`

### Phase 1 — K-Means Clustering (k=5)
- Features: size, bedrooms, floor, exterior exposure, property type
- Validated with elbow method + silhouette score
- Segments: **Entry-Level Interior**, **High-Rise Exterior**, **Grand Estate**, **Standard Exterior Living**, **Urban Premium**

### Phase 2 — Association Analysis
- Apriori algorithm applied to property segments
- Evaluated by support, confidence, and lift

### Phase 3 — OLS Linear Regression
- VIF-based multicollinearity filtering (threshold = 10)
- RFECV feature selection (Repeated K-Fold, 5 splits × 3 repeats)
- Statsmodels OLS for p-values and 95% prediction intervals
- Target: monthly rent in €

### Phase 4 — Logistic Regression
- Same VIF pipeline + stratified train/test split
- Binary target: High Rent (≥ €1,800)
- Evaluated via AUC-ROC, confusion matrix, odds ratios, and threshold sensitivity

---

## Tech Stack

| Layer | Tools |
|---|---|
| **App Framework** | [Streamlit](https://streamlit.io/), streamlit-extras |
| **Data** | pandas, numpy, openpyxl |
| **Machine Learning** | scikit-learn (KMeans, RFECV, train_test_split) |
| **Statistics** | statsmodels (OLS, Logit, VIF) |
| **Visualization** | Plotly Express, Plotly Graph Objects |

---

## Project Structure

```
MadridRental/
├── madrid_rental_app.py      # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
└── data/
    └── Houses for rent in Madrid.xlsx   # ~2,100 Idealista listings
```

---

## Dataset

- **Source:** Idealista (Madrid rental listings)
- **Size:** 2,100 raw → 2,089 after cleaning
- **Raw columns:** Rent, Area (Sq.Mt), Bedrooms, Floor, Outer, Elevator, District, Address, Penthouse, Cottage, Duplex, Semidetached
- **Engineered columns:** `Is_Special`, `Price_per_sqm`, `SqMt_per_Bed`, `District_Premium`, `Zone`, `Is_Central`, `Is_Studio`, `High_Rent`, `Cluster`, `Segment`

> The dataset is included in this repository under `data/`. It is used for academic purposes only.

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/mariangarabana/MadridRental.git
cd MadridRental
```

**2. Create and activate a virtual environment** *(recommended)*
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run madrid_rental_app.py
```

The app will open at `http://localhost:8501`.

---

## Course Info

| | |
|---|---|
| **Course** | Machine Learning I |
| **Program** | Master in Big Data & Business Analytics (MBDS) |
| **Institution** | IE University |
| **Year** | 2026 |
