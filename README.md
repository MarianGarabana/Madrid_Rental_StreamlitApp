Streamlit App Link: https://madridrentalapp-mariangarabana.streamlit.app/

# Madrid Rental Market — Interactive ML Dashboard
**Machine Learning I · MBDS · IE University · 2026**

A Streamlit application that brings the full Madrid rental housing analysis to life — from raw data exploration to live rent predictions and high-rent classification.

---

## Live App

> 🚀 **[Launch App on Streamlit Cloud]([https://your-app-url.streamlit.app](https://madridrentalapp-mariangarabana.streamlit.app/))**

---

## What It Does

The app has four interactive pages:

| Page | What you can do |
|---|---|
| 🔍 Market Explorer | Filter listings by district and rent range. See distributions, median rents by district, rent vs. size scatter, and a correlation heatmap |
| 🏘️ Property Segments | Explore the 5 K-Means clusters and their profiles. Enter a property's details to find which segment it belongs to |
| 💶 Rent Predictor | See OLS model performance (R², RMSE, MAE), coefficient effects, and predict the monthly rent of any property with a 95% confidence interval |
| 📊 High Rent Classifier | Explore the logit model's ROC curve, confusion matrix, odds ratios, and classify any property as High Rent (≥ €1,800) with a probability gauge |

All models are trained at startup from the raw data — no pre-saved model files needed.

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/MarianGarabana/Madrid-Rental-Analysis_ML.git
cd Madrid-Rental-Analysis_ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
mkdir data
# Place "Houses for rent in Madrid.xlsx" inside the data/ folder

# 4. Run the app
streamlit run madrid_rental_app.py
```

---

## Project Structure

```
📁 Madrid-Rental-Analysis_ML/
│
├── madrid_rental_app.py                              # Streamlit app (this file)
├── data/
│   └── Houses for rent in Madrid.xlsx               # Dataset (not in repo)
├── EDA_Group_Assignment_Formatted.ipynb              # Phase 0 — EDA & cleaning
├── Phase1_Phase2_Segmentation_Association_0225.ipynb # Phase 1/2 — Clustering & Apriori
├── Phase3_Linear_Regression_0225.ipynb               # Phase 3 — OLS Regression
├── Phase4_Logit_Regression_0225.ipynb                # Phase 4 — Logit Classification
├── requirements.txt
└── README.md
```

---

## Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
statsmodels
openpyxl
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## ML Pipeline

The app reproduces the full group assignment pipeline:

**Phase 0 — EDA & Cleaning:** 2,100 listings → 2,089 after dropping missing values and fixing a floor outlier. Feature engineering includes `Is_Special`, `Price_per_sqm`, `District_Premium`, `Zone`, `Is_Central`, `Is_Studio`, and the binary target `High_Rent`.

**Phase 1 — K-Means Clustering (k=5):** Clusters on 5 structural variables (Sq.Mt, Bedrooms, Floor, Is_Special, Outer), validated by elbow method and silhouette score. Segments: Entry-Level Interior, High-Rise Exterior, Grand Estate, Standard Exterior Living, Urban Premium.

**Phase 2 — Association Analysis:** Apriori algorithm on key segments, evaluated by support, confidence, and lift (see dedicated notebook).

**Phase 3 — OLS Linear Regression:** VIF-based multicollinearity removal + RFECV feature selection. Final model via statsmodels for p-values and prediction intervals. Target: monthly rent (€).

**Phase 4 — Logistic Regression:** Same VIF pipeline + stratified train/test split. Binary target: High Rent (≥ €1,800). Evaluated with ROC-AUC, confusion matrix, odds ratios, and threshold sensitivity table.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (dataset excluded)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `madrid_rental_app.py`
5. Upload the dataset via **Secrets** or use a hosted file path

> ⚠️ The dataset is not included in this repo due to data source restrictions (idealista.com). To run the app you need the original `Houses for rent in Madrid.xlsx` file placed in a `data/` folder.

---

*Built by Marian Garabana · MBDS 2026 · IE School of Human Sciences & Technology*
