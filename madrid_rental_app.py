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


# ══════════════════════════════════════════════════════════════════════════════
# APP CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Madrid Rental Market", page_icon="🏠", layout="wide")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING  ← from EDA notebook
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
    df['Is_Special'] = ((df['Penthouse'] == 1) | (df['Cottage'] == 1) | (df['Duplex'] == 1) | (df['Semidetached'] == 1)).astype(int)

    # Price per m² — for profiling/display only (DO NOT use as model input, leaks Rent)
    df['Price_per_sqm'] = (df['Rent'] / df['Sq.Mt']).round(2)

    # m² per bedroom — spaciousness proxy
    df['SqMt_per_Bed'] = (df['Sq.Mt'] / df['Bedrooms'].replace(0, np.nan)).round(1)

    # District quality proxy
    district_median     = df.groupby('District')['Rent'].median()
    df['District_Premium'] = df['District'].map(district_median)

    # Geographic zone grouping
    zone_map = {
        'Salamanca': 'Prime Center',     'Chamberí': 'Prime Center',     'Retiro': 'Prime Center',
        'Centro': 'Historic Core',
        'Chamartín': 'Business North',   'Tetuán': 'Business North',
        'Moncloa': 'University/West',    'Arganzuela': 'University/West',
        'Hortaleza': 'Suburban Premium', 'Fuencarral': 'Suburban Premium','Barajas': 'Suburban Premium',
        'Ciudad Lineal': 'Suburban Standard','San Blás': 'Suburban Standard','Moratalaz': 'Suburban Standard',
        'Latina': 'Working Class South', 'Carabanchel': 'Working Class South','Usera': 'Working Class South',
        'Puente Vallecas': 'Working Class South','Villa de Vallecas': 'Working Class South',
        'Vicálvaro': 'Working Class South',
    }
    df['Zone']       = df['District'].map(zone_map)
    df['Is_Central'] = df['Zone'].isin(['Prime Center', 'Historic Core']).astype(int)
    df['Is_Studio']  = df['Address'].str.contains('Estudio', case=False, na=False).astype(int)

    # Binary target for Phase 4 classifier
    df['High_Rent']  = (df['Rent'] >= 1800).astype(int)

    return df


df = load_and_clean_data()


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
    scaler_r      = StandardScaler()
    X_train_s_r   = scaler_r.fit_transform(X_train_vif_r)
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
    X_ols_test  = sm.add_constant(X_test_r[selected_features])
    y_pred_r    = ols_model.predict(X_ols_test)
    r2_test_r   = r2_score(y_test_r, y_pred_r)
    rmse_r      = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
    mae_r       = float(mean_absolute_error(y_test_r, y_pred_r))

    # Coefficient table for display
    coef_df = pd.DataFrame({
        'Feature':        ols_model.params.index,
        'Effect (€)':     ols_model.params.values.round(1),
        'p-value':        ols_model.pvalues.values.round(4),
        'CI Low (€)':     ols_model.conf_int()[0].values.round(1),
        'CI High (€)':    ols_model.conf_int()[1].values.round(1),
        'Significant':    ['✓' if p < 0.05 else '✗' for p in ols_model.pvalues],
    })

    results.update({
        'ols_model':        ols_model,
        'selected_features':selected_features,
        'r2_train_r':       ols_model.rsquared,
        'r2_test_r':        r2_test_r,
        'rmse_r':           rmse_r,
        'mae_r':            mae_r,
        'y_test_r':         y_test_r,
        'y_pred_r':         y_pred_r,
        'coef_df':          coef_df,
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

    X_sm_train = sm.add_constant(X_train_vif_l)
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
    ci_exp      = np.exp(logit_model.conf_int())
    odds_table  = pd.DataFrame({
        'Exp(B)':       np.exp(logit_model.params).round(4),
        'p-value':      logit_model.pvalues.round(4),
        'CI Low':       ci_exp[0].round(4),
        'CI High':      ci_exp[1].round(4),
        'Significant':  ['✓' if p < 0.05 else '✗' for p in logit_model.pvalues],
    })

    results.update({
        'logit_model':   logit_model,
        'logit_features':logit_features,
        'auc_test_l':    auc_test_l,
        'auc_train_l':   auc_train_l,
        'acc_l':         acc_l,
        'cm_l':          cm_l,
        'fpr_test':      fpr_test,
        'tpr_test':      tpr_test,
        'fpr_train':     fpr_train,
        'tpr_train':     tpr_train,
        'y_test_l':      y_test_l,
        'y_prob_l':      y_prob_l,
        'threshold_df':  threshold_df,
        'odds_table':    odds_table,
    })

    return results


# Train everything — shows spinner on first load, cached after that
with st.spinner("Training models…"):
    M = train_all_models(df)

# Attach cluster labels to main dataframe
df['Cluster'] = M['cluster_labels']
df['Segment'] = df['Cluster'].map(lambda c: M['segment_labels'][c][0])


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
# TODO: style the sidebar however you like (logo, colours, etc.)

page = st.sidebar.radio("Navigate", [
    "🔍 Market Explorer",
    "🏘️ Property Segments",
    "💶 Rent Predictor",
    "📊 High Rent Classifier",
])

st.title(page)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Market Explorer":
    rent_min, rent_max = int(df['Rent'].min()), int(df['Rent'].max())
    selected_rent = st.sidebar.slider("Rent Range (€)", rent_min, rent_max, (rent_min, rent_max))
    selected_districts = st.sidebar.multiselect("Select Districts:", df['District'].unique().tolist(), default= df['District'].unique().tolist())

    filtered = df[df['District'].isin(selected_districts) & df['Rent'].between(selected_rent[0], selected_rent[1])]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
            st.metric("Number of Properties", len(filtered))
    with col2:
        st.metric("Median Rent", filtered['Rent'].median())
    with col3:
        st.metric("Avg Square Meters", filtered['Sq.Mt'].mean().round(2))
    with col4:
        st.metric("Avg Price per m2", filtered['Price_per_sqm'].mean().round(2))

    fig = px.histogram(filtered, x='Rent', nbins=40)
    fig.add_vline(x=filtered['Rent'].median(), line_width=2, line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(filtered.groupby('District')['Rent'].median().reset_index().sort_values('Rent'), x='Rent', y='District', orientation='h')
    fig.update_layout(title='Median Rent by District')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(filtered, x='Sq.Mt', y='Rent', color='District', size='Rent', title='Rent vs Sq.Mt')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.imshow(filtered[['Rent','Sq.Mt','Bedrooms','Floor','Outer','Elevator','Price_per_sqm']].corr(), text_auto='.2f', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PROPERTY SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ Property Segments":

    summary = df.groupby('Segment').agg(
        Properties=('Rent', 'count'),
        Median_rent=('Rent', 'median'),
        Median_SqMt=('Sq.Mt', 'median'),
        Median_Floor=('Floor', 'median'),
        Pct_Outer=('Outer', 'mean'),
    )

    st.dataframe(summary)

    fig = px.scatter(df, x='Sq.Mt', y='Rent', color='Segment', size='Price_per_sqm', title='Rent vs Sq.Mt')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(summary, x='Segment', y='Median_rent', orientation='v', title='Median Rent by Segment')
    st.plotly_chart(fig, use_container_width=True)


    for cluster_id, (name, desc) in M['segment_labels'].items():
        with st.expander(f"{name}"):
            st.write(desc)
    
    sqm = st.number_input("Square meters", min_value=0, max_value=1000, value=100)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
    floor = st.number_input("Floor", min_value=0, max_value=10, value=1)
    is_special = st.checkbox("Special", value=False)
    outer = st.checkbox("Outer", value=False)

    if st.button("Classify"):
        arr    = np.array([[sqm, bedrooms, floor, int(is_special), int(outer)]])
        scaled = M['cluster_scaler'].transform(arr)
        cid    = M['kmeans'].predict(scaled)[0]
        name, desc = M['segment_labels'][cid]
        st.write(f"This property belongs to the {name} segment")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RENT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💶 Rent Predictor":

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Test", f"{M['r2_test_r']:.2f}")
    with col2:
        st.metric("RMSE", f"{M['rmse_r']:.2f}")
    with col3:
        st.metric("MAE", f"{M['mae_r']:.2f}")

    coef_plot = M['coef_df'][M['coef_df']['Feature'] != 'const'].copy()
    coef_plot['Direction'] = coef_plot['Effect (€)'].apply(lambda x: 'Increases rent' if x > 0 else 'Decreases rent')
    fig = px.bar(coef_plot, x='Effect (€)', y='Feature', color='Direction', orientation='h',
                 color_discrete_map={'Increases rent': 'green', 'Decreases rent': 'red'},
                 title='Effect of Each Feature on Rent (€)')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(x=M['y_test_r'], y=M['y_pred_r'],
                     labels={'x': 'Actual Rent (€)', 'y': 'Predicted Rent (€)'},
                     title='Actual vs Predicted Rent')
    min_v, max_v = float(M['y_test_r'].min()), float(M['y_test_r'].max())
    fig.add_shape(type='line', x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                  line=dict(color='red', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predict rent for a new property")
    col_a, col_b = st.columns(2)
    with col_a:
        sqm_r      = st.number_input("Size (m²)",    15,  500, 80,  key='r_sqm')
        bedrooms_r = st.number_input("Bedrooms",      0,    8,  2,  key='r_bed')
        floor_r    = st.number_input("Floor",          0,   25,  3,  key='r_floor')
        outer_r    = st.checkbox("Exterior facing",            key='r_outer')
    with col_b:
        elevator_r   = st.checkbox("Has elevator",             key='r_elev')
        is_special_r = st.checkbox("Penthouse / Duplex / Cottage", key='r_spec')
        is_central_r = st.checkbox("Central district",         key='r_cent')
        is_studio_r  = st.checkbox("Listed as Studio",         key='r_stud')

    if st.button("Predict Rent", type="primary"):
        input_dict = {
            'Sq.Mt': sqm_r, 'Bedrooms': bedrooms_r, 'Floor': floor_r,
            'Outer': int(outer_r), 'Elevator': int(elevator_r),
            'Is_Special': int(is_special_r), 'Is_Central': int(is_central_r),
            'Is_Studio': int(is_studio_r)
        }
        input_df = pd.DataFrame([input_dict])[M['selected_features']]
        input_sm = sm.add_constant(input_df, has_constant='add')
        prediction = M['ols_model'].predict(input_sm)[0]
        st.metric("Predicted Monthly Rent", f"€{prediction:,.0f}")
        pred_sum = M['ols_model'].get_prediction(input_sm).summary_frame(alpha=0.05)
        lo = pred_sum['obs_ci_lower'].values[0]
        hi = pred_sum['obs_ci_upper'].values[0]
        st.caption(f"95% prediction interval: €{lo:,.0f} – €{hi:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HIGH RENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 High Rent Classifier":

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{M['acc_l']:.2f}")
    with col2:
        st.metric("AUC Test", f"{M['auc_test_l']:.2f}")
    with col3:
        st.metric("AUC Gap", f"{M['auc_train_l'] - M['auc_test_l']:.2f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=M['fpr_train'], y=M['tpr_train'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=M['fpr_test'], y=M['tpr_test'], mode='lines', name='Test'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Baseline'))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.imshow(M['cm_l'], text_auto=True, x=['Pred Low','Pred High'], y=['Actual Low','Actual High'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(M['odds_table'])

    prob_low  = M['y_prob_l'][M['y_test_l'] == 0]
    prob_high = M['y_prob_l'][M['y_test_l'] == 1]
    
    fig = px.histogram(x=prob_low, nbins=20, color_discrete_sequence=['blue'])
    fig.add_histogram(x=prob_high, nbinsx=20, marker_color='red', opacity=0.6, name='High Rent')
    fig.add_vline(x=0.5, line_color='black', line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Is this property High Rent (≥ €1,800)?")
    col_a, col_b = st.columns(2)
    with col_a:
        sqm_l      = st.number_input("Size (m²)",    15,  500, 80,  key='l_sqm')
        bedrooms_l = st.number_input("Bedrooms",      0,    8,  2,  key='l_bed')
        floor_l    = st.number_input("Floor",          0,   25,  3,  key='l_floor')
        outer_l    = st.checkbox("Exterior facing",            key='l_outer')
    with col_b:
        elevator_l   = st.checkbox("Has elevator",             key='l_elev')
        is_special_l = st.checkbox("Penthouse / Duplex / Cottage", key='l_spec')
        is_central_l = st.checkbox("Central district",         key='l_cent')
        is_studio_l  = st.checkbox("Listed as Studio",         key='l_stud')

    if st.button("Classify", type="primary"):
        input_dict = {
            'Sq.Mt': sqm_l, 'Bedrooms': bedrooms_l, 'Floor': floor_l,
            'Outer': int(outer_l), 'Elevator': int(elevator_l),
            'Is_Special': int(is_special_l), 'Is_Central': int(is_central_l),
            'Is_Studio': int(is_studio_l)
        }
        input_df = pd.DataFrame([input_dict])[M['logit_features']]
        input_sm = sm.add_constant(input_df, has_constant='add')
        probability = float(M['logit_model'].predict(input_sm)[0])
        label = "🔴 High Rent" if probability >= 0.5 else "🟢 Low Rent"
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classification", label)
        with col2:
            st.metric("Probability of High Rent", f"{probability:.1%}")
        gauge_chart = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': '%'},
            title={'text': "Probability of High Rent"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#e74c3c' if probability >= 0.5 else '#27ae60'},
                'steps': [{'range': [0, 50],  'color': '#eafaf1'},
                           {'range': [50, 100], 'color': '#fdedec'}],
                'threshold': {'line': {'color': 'black', 'width': 3}, 'value': 50}
            }
        ))
        st.plotly_chart(gauge_chart, use_container_width=True)

    threshold_sensitivity_table = M['threshold_df'].round(2)
    st.dataframe(threshold_sensitivity_table)

