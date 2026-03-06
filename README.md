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
| **Market Explorer** | Filter listings by district and rent range. Three tabs: **Charts** (rent distribution histogram, box plots by district, rent vs. size scatter, correlation heatmap); **By Zone** (median rent by geographic zone bar chart, rent distribution by zone box plot); **Raw Data** (filterable table with CSV download). |
| **Property Segments** | Explore 5 K-Means clusters. Overview tab shows segment expanders, share of listings donut chart, median rent bar chart, rent vs. size scatter, and a normalised radar chart comparing all segments. Classify tab predicts the segment of any property. |
| **Association Rules** | Apriori algorithm applied to binary property attributes and market segment labels. Interactive filters for confidence, lift, support, confidence difference, confidence ratio, and rule complexity (antecedents/consequents). Top rules visualised as a horizontal bar chart and a support vs. confidence scatter. |
| **Rent Predictor** | OLS linear regression with VIF filtering and RFECV feature selection. Predict monthly rent with a 95% prediction interval and see where it sits in the market distribution. Performance tab includes coefficient chart, actual vs. predicted scatter, residual plot, and Q-Q normality check. |
| **High Rent Classifier** | Logistic regression with interactive probability threshold. Performance tab shows ROC curve (train + test), confusion matrix, probability separation histogram, threshold sensitivity table, and odds ratios. Classify tab outputs a probability gauge for any property. |

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
| **Machine Learning** | scikit-learn (KMeans, RFECV, train_test_split), mlxtend (Apriori, association_rules) |
| **Statistics** | statsmodels (OLS, Logit, VIF), scipy (Q-Q normality test) |
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

## AI Usage

[Claude Sonnet](https://www.anthropic.com/claude) (Anthropic) was used as an AI assistant during development for:
- **UI/UX design** — color scheme decisions, chart layout, and visual coherence across pages
- **Bug fixing** — resolving color mapping inconsistencies in Plotly charts and ensuring consistent segment colors across all visualisations

All ML modelling, data analysis, and application logic were written and validated by the author.

---

## Course Info

| | |
|---|---|
| **Course** | Machine Learning I |
| **Program** | Master in Big Data & Business Analytics (MBDS) |
| **Institution** | IE University |
| **Year** | 2026 |
