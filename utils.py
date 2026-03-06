import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             roc_auc_score, accuracy_score, confusion_matrix,
                             roc_curve, precision_score, recall_score, f1_score)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.chart_container import chart_container
from streamlit_extras.add_vertical_space import add_vertical_space
from mlxtend.frequent_patterns import apriori, association_rules


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
.stApp { background-color: #F4F4F4; }
[data-testid="stSidebar"] {
    background-color: #E2F46E;
}
[data-testid="stSidebar"] * {
    color: #333333 !important;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border-left: 4px solid #B72683;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #333333 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600;
}
h1, h2, h3 { color: #333333; }
hr { border-color: #D9D9D9; }
.stButton > button {
    background-color: #B72683;
    color: #FFFFFF;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #9A1F6E;
    color: #FFFFFF;
}
.chart-tooltip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 1.5px solid #B72683;
    color: #B72683;
    font-size: 10px;
    font-weight: 700;
    cursor: help;
    line-height: 1;
    flex-shrink: 0;
}
.chart-tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    background: #333333;
    color: #ffffff;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 400;
    width: 280px;
    white-space: normal;
    z-index: 9999;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s;
    text-align: left;
}
.chart-tooltip:hover::after {
    opacity: 1;
}
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border-radius: 8px;
}
[data-testid="stSidebarNav"] {
    display: none;
}
[data-testid="stPageLink"] {
    background-color: #C8DC58;
    border-radius: 6px;
    margin-bottom: 2px;
}
</style>
"""


def apply_css():
    st.markdown(_CSS, unsafe_allow_html=True)
    style_metric_cards(background_color="#FFFFFF", border_left_color="#B72683", border_radius_px=8)


def render_sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:12px 0 8px 0;'>
            <span style='font-size:2.2rem;'>🏠</span>
            <h2 style='margin:4px 0 0 0; color:#B72683; font-size:1.05rem; font-weight:700;'>
                Madrid Rental Market
            </h2>
            <p style='font-size:0.73rem; color:#555555; margin:2px 0 0 0;'>
                IE University · MBDS · 2025
            </p>
        </div>
        <hr style='border-color:#333333; opacity:0.2; margin:8px 0 16px 0;'>
        """, unsafe_allow_html=True)

        st.page_link("Home.py",                              label="Home")
        st.page_link("pages/1_Market_Explorer.py",           label="Market Explorer")
        st.page_link("pages/2_Property_Segments.py",         label="Property Segments")
        st.page_link("pages/3_Association_Rules.py",         label="Association Rules")
        st.page_link("pages/4_Rent_Predictor.py",            label="Rent Predictor")
        st.page_link("pages/5_High_Rent_Classifier.py",      label="High Rent Classifier")

        st.markdown(f"""
        <hr style='border-color:#333333; opacity:0.2; margin:16px 0 8px 0;'>
        <p style='font-size:0.70rem; color:#555555; text-align:center; line-height:1.6;'>
            Dataset: Idealista Madrid<br>
            n = {len(df):,} listings<br>
            Built with Streamlit · scikit-learn
        </p>
        """, unsafe_allow_html=True)


def chart_header(title, description):
    desc_esc = description.replace('"', '&quot;')
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">'
        f'<span style="font-weight:600;font-size:1.05rem;">{title}</span>'
        f'<span class="chart-tooltip" data-tooltip="{desc_esc}">?</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_clean_data():
    df = pd.read_excel("data/Houses for rent in Madrid.xlsx")

    # Drop column with 64% missing values
    df = df.drop(columns='Number')

    # Drop rows where Area is missing (only 0.2% — safe to drop)
    df = df.dropna(subset=['Area'])

    # Fix Floor outlier (value > 100 was a data entry typo)
    df.loc[df['Floor'] > 100, 'Floor'] = None
    df['Floor']    = df['Floor'].fillna(df['Floor'].median())
    df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
    df['Outer']    = df['Outer'].fillna(df['Outer'].mode()[0])
    df['Elevator'] = df['Elevator'].fillna(df['Elevator'].mode()[0])

    # ── Feature engineering from Phase 1 ─────────────────────────────────

    # Is_Special: rare property types combined into one meaningful flag
    df['Is_Special'] = ((df['Penthouse'] == 1) | (df['Cottage'] == 1) |
                        (df['Duplex'] == 1)    | (df['Semidetached'] == 1)).astype(int)

    # Price per m² — for profiling/display only (DO NOT use as model input, leaks Rent)
    df['Price_per_sqm'] = (df['Rent'] / df['Sq.Mt']).round(2)

    # m² per bedroom — spaciousness proxy
    df['SqMt_per_Bed'] = (df['Sq.Mt'] / df['Bedrooms'].replace(0, np.nan)).round(1)

    # District quality proxy
    district_median        = df.groupby('District')['Rent'].median()
    df['District_Premium'] = df['District'].map(district_median)

    # Geographic zone grouping
    zone_map = {
        'Salamanca': 'Prime Center',     'Chamberí': 'Prime Center',     'Retiro': 'Prime Center',
        'Centro': 'Historic Core',
        'Chamartín': 'Business North',   'Tetuán': 'Business North',
        'Moncloa': 'University/West',    'Arganzuela': 'University/West',
        'Hortaleza': 'Suburban Premium', 'Fuencarral': 'Suburban Premium', 'Barajas': 'Suburban Premium',
        'Ciudad Lineal': 'Suburban Standard', 'San Blás': 'Suburban Standard', 'Moratalaz': 'Suburban Standard',
        'Latina': 'Working Class South', 'Carabanchel': 'Working Class South', 'Usera': 'Working Class South',
        'Puente Vallecas': 'Working Class South', 'Villa de Vallecas': 'Working Class South',
        'Vicálvaro': 'Working Class South',
    }
    df['Zone']       = df['District'].map(zone_map)
    df['Is_Central'] = df['Zone'].isin(['Prime Center', 'Historic Core']).astype(int)
    df['Is_Studio']  = df['Address'].str.contains('Estudio', case=False, na=False).astype(int)

    # Binary target for Phase 4 classifier
    df['High_Rent'] = (df['Rent'] >= 1800).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# VIF HELPER  ← shared by Phase 3 and Phase 4
# ══════════════════════════════════════════════════════════════════════════════

def remove_vif(df_in, thresh=10.0):
    """Iteratively drop the highest-VIF variable until all VIF < threshold."""
    df_w = df_in.copy()
    while True:
        vifs = pd.Series(
            [variance_inflation_factor(df_w.values, i) for i in range(df_w.shape[1])],
            index=df_w.columns
        )
        if vifs.max() <= thresh:
            break
        df_w = df_w.drop(columns=vifs.idxmax())
    return df_w


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING  ← all three models trained once at startup
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def train_all_models(_df):
    """Train K-Means, OLS, and Logit. Underscore prefix stops Streamlit hashing _df."""

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — K-Means Clustering  (k=5, validated by elbow + silhouette)
    # ─────────────────────────────────────────────────────────────────────────
    clustering_vars = ['Sq.Mt', 'Bedrooms', 'Floor', 'Is_Special', 'Outer']
    X_clust = _df[clustering_vars].copy()

    cluster_scaler = StandardScaler()
    X_scaled       = cluster_scaler.fit_transform(X_clust)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    cluster_assignments = kmeans.predict(X_scaled)

    # Business labels validated against full multi-variable profiling from Phase 1.
    # Key anchors:
    #   Entry-Level Interior  → ~0% Outer, lowest price/m²
    #   High-Rise Exterior    → median floor ≥ 7
    #   Grand Estate          → median Sq.Mt > 300 (cottages/villas)
    #   Standard Exterior     → largest cluster (~50% of listings)
    #   Urban Premium         → high Is_Special %, penthouses/duplexes
    segment_labels = {
        0: ('Entry-Level Interior',
            'Small interior flats on low floors. No exterior views. Lowest price/m² — pure budget segment.'),
        1: ('High-Rise Exterior',
            'Standard flats on high floors (median ≥ 7) with exterior exposure. '
            'The height premium: better light and views justify a €200–300 uplift.'),
        2: ('Grand Estate',
            'Large villas, cottages, and chalets. Median >300 m². '
            'The luxury of space rather than height — often ground/low floor.'),
        3: ('Standard Exterior Living',
            'Market volume driver (~50% of listings). Mid-size exterior flats on lower floors. '
            'Bread-and-butter Madrid rental stock.'),
        4: ('Urban Premium',
            'Penthouses and duplexes. Elevated price/m², concentrated in prime districts. Status properties.'),
    }

    results.update({
        'kmeans':          kmeans,
        'cluster_scaler':  cluster_scaler,
        'clustering_vars': clustering_vars,
        'cluster_labels':  cluster_assignments,
        'segment_labels':  segment_labels,
    })

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Linear Regression  (OLS + VIF + RFECV, statsmodels)
    # ─────────────────────────────────────────────────────────────────────────
    features = ['Sq.Mt', 'Bedrooms', 'Floor', 'Outer', 'Elevator',
                'Is_Special', 'Is_Central', 'Is_Studio']

    X_r = _df[features].copy()
    y_r = _df['Rent'].copy()

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_r, y_r, test_size=0.2, random_state=42
    )

    # VIF removal to fix multicollinearity
    X_train_vif_r = remove_vif(X_train_r, thresh=10.0)

    # RFECV: finds optimal feature subset via cross-validation
    scaler_r    = StandardScaler()
    X_train_s_r = scaler_r.fit_transform(X_train_vif_r)
    rfecv_r = RFECV(
        estimator=LinearRegression(), step=1,
        cv=RepeatedKFold(n_splits=5, n_repeats=3),
        scoring='neg_mean_squared_error',
        min_features_to_select=1
    )
    rfecv_r.fit(X_train_s_r, y_train_r)
    selected_features = X_train_vif_r.columns[rfecv_r.support_].tolist()

    # Final OLS — statsmodels gives p-values, CIs, and prediction intervals
    X_ols_train = sm.add_constant(X_train_vif_r[selected_features])
    ols_model   = sm.OLS(y_train_r, X_ols_train).fit()

    # Test evaluation
    X_ols_test = sm.add_constant(X_test_r[selected_features])
    y_pred_r   = ols_model.predict(X_ols_test)
    r2_test_r  = r2_score(y_test_r, y_pred_r)
    rmse_r     = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
    mae_r      = float(mean_absolute_error(y_test_r, y_pred_r))

    # Coefficient table for display
    coef_df = pd.DataFrame({
        'Feature':     ols_model.params.index,
        'Effect (€)':  ols_model.params.values.round(1),
        'p-value':     ols_model.pvalues.values.round(4),
        'CI Low (€)':  ols_model.conf_int()[0].values.round(1),
        'CI High (€)': ols_model.conf_int()[1].values.round(1),
        'Significant': ['✓' if p < 0.05 else '✗' for p in ols_model.pvalues],
    })

    results.update({
        'ols_model':         ols_model,
        'selected_features': selected_features,
        'r2_train_r':        ols_model.rsquared,
        'r2_test_r':         r2_test_r,
        'rmse_r':            rmse_r,
        'mae_r':             mae_r,
        'y_test_r':          y_test_r,
        'y_pred_r':          y_pred_r,
        'coef_df':           coef_df,
    })

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4 — Logit Classification  (High Rent ≥ €1,800)
    # ─────────────────────────────────────────────────────────────────────────
    X_l = _df[features].copy()
    y_l = _df['High_Rent'].copy()

    # Stratified split preserves class balance
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_l, y_l, test_size=0.2, random_state=42, stratify=y_l
    )

    X_train_vif_l  = remove_vif(X_train_l, thresh=10.0)
    logit_features = X_train_vif_l.columns.tolist()

    X_sm_train  = sm.add_constant(X_train_vif_l)
    logit_model = sm.Logit(y_train_l, X_sm_train).fit(disp=0)

    # Test predictions
    X_sm_test    = sm.add_constant(X_test_l[logit_features])
    y_prob_l     = logit_model.predict(X_sm_test)
    y_pred_l     = (y_prob_l >= 0.5).astype(int)
    y_prob_train = logit_model.predict(X_sm_train)

    # Metrics
    auc_test_l  = float(roc_auc_score(y_test_l, y_prob_l))
    auc_train_l = float(roc_auc_score(y_train_l, y_prob_train))
    acc_l       = float(accuracy_score(y_test_l, y_pred_l))
    cm_l        = confusion_matrix(y_test_l, y_pred_l)

    # ROC curves (test + train, for overfit check)
    fpr_test,  tpr_test,  _ = roc_curve(y_test_l,  y_prob_l)
    fpr_train, tpr_train, _ = roc_curve(y_train_l, y_prob_train)

    # Threshold sensitivity table
    thresh_rows = []
    for cutoff in [0.3, 0.4, 0.5, 0.6, 0.7]:
        yp = (y_prob_l >= cutoff).astype(int)
        thresh_rows.append({
            'Cutoff':      cutoff,
            'Accuracy':    round(accuracy_score(y_test_l, yp), 4),
            'Precision':   round(precision_score(y_test_l, yp, zero_division=0), 4),
            'Recall':      round(recall_score(y_test_l, yp, zero_division=0), 4),
            'Specificity': round(recall_score(y_test_l, yp, pos_label=0, zero_division=0), 4),
            'F1':          round(f1_score(y_test_l, yp, zero_division=0), 4),
        })
    threshold_df = pd.DataFrame(thresh_rows)

    # Odds ratios table
    ci_exp     = np.exp(logit_model.conf_int())
    odds_table = pd.DataFrame({
        'Exp(B)':      np.exp(logit_model.params).round(4),
        'p-value':     logit_model.pvalues.round(4),
        'CI Low':      ci_exp[0].round(4),
        'CI High':     ci_exp[1].round(4),
        'Significant': ['✓' if p < 0.05 else '✗' for p in logit_model.pvalues],
    })

    results.update({
        'logit_model':    logit_model,
        'logit_features': logit_features,
        'auc_test_l':     auc_test_l,
        'auc_train_l':    auc_train_l,
        'acc_l':          acc_l,
        'cm_l':           cm_l,
        'fpr_test':       fpr_test,
        'tpr_test':       tpr_test,
        'fpr_train':      fpr_train,
        'tpr_train':      tpr_train,
        'y_test_l':       y_test_l,
        'y_prob_l':       y_prob_l,
        'threshold_df':   threshold_df,
        'odds_table':     odds_table,
    })

    return results
