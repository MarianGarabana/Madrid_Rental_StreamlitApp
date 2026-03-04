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
# APP CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Madrid Rental Market", page_icon="🏠", layout="wide")


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #FDF6EC;
}
[data-testid="stSidebar"] * {
    color: #2C3E50 !important;
}
[data-testid="stMetric"] {
    background-color: #FDF6EC;
    border-left: 4px solid #C0392B;
    border-radius: 6px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #2C3E50 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600;
}
h1, h2, h3 { color: #2C3E50; }
hr { border-color: #E8D5B7; }
</style>
""", unsafe_allow_html=True)


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
    df['Is_Special'] = ((df['Penthouse'] == 1) | (df['Cottage'] == 1) |
                        (df['Duplex'] == 1)    | (df['Semidetached'] == 1)).astype(int)

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

style_metric_cards(background_color="#FDF6EC", border_left_color="#C0392B", border_radius_px=6)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 8px 0;'>
        <span style='font-size:2.2rem;'>🏠</span>
        <h2 style='margin:4px 0 0 0; color:#C0392B; font-size:1.05rem; font-weight:700;'>
            Madrid Rental Market
        </h2>
        <p style='font-size:0.73rem; color:#95a5a6; margin:2px 0 0 0;'>
            IE University · MBDS · 2025
        </p>
    </div>
    <hr style='border-color:#E8D5B7; margin:8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🔍 Market Explorer",
        "🏘️ Property Segments",
        "🔗 Association Rules",
        "💶 Rent Predictor",
        "📊 High Rent Classifier",
    ])

    st.markdown(f"""
    <hr style='border-color:#E8D5B7; margin:16px 0 8px 0;'>
    <p style='font-size:0.70rem; color:#bdc3c7; text-align:center; line-height:1.6;'>
        Dataset: Idealista Madrid<br>
        n = {len(df):,} listings<br>
        Built with Streamlit · scikit-learn
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title(page)
st.caption("Marian Garabana · MBDS Student at IE University")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Market Explorer":
    rent_min, rent_max = int(df['Rent'].min()), int(df['Rent'].max())
    selected_rent = st.sidebar.slider("Rent Range (€)", rent_min, rent_max, (rent_min, rent_max))
    selected_districts = st.sidebar.multiselect(
        "Select Districts:", df['District'].unique().tolist(),
        default=df['District'].unique().tolist()
    )

    filtered = df[
        df['District'].isin(selected_districts) &
        df['Rent'].between(selected_rent[0], selected_rent[1])
    ]

    # Metrics with delta vs. full dataset baseline
    overall_median = df['Rent'].median()
    overall_sqmt   = df['Sq.Mt'].mean()
    overall_ppsqm  = df['Price_per_sqm'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Properties",   f"{len(filtered):,}",
                delta=f"{len(filtered) - len(df):,} vs all")
    col2.metric("Median Rent",  f"€{filtered['Rent'].median():,.0f}",
                delta=f"€{filtered['Rent'].median() - overall_median:+,.0f} vs overall")
    col3.metric("Avg Size",     f"{filtered['Sq.Mt'].mean():.0f} m²",
                delta=f"{filtered['Sq.Mt'].mean() - overall_sqmt:+.0f} m²")
    col4.metric("Avg Price/m²", f"€{filtered['Price_per_sqm'].mean():.2f}",
                delta=f"€{filtered['Price_per_sqm'].mean() - overall_ppsqm:+.2f}")

    with st.expander("How this works"):
        st.write(
            f"Exploratory analysis of {len(df):,} Madrid rental listings sourced from Idealista. "
            "Use the sidebar to filter by rent range and district. "
            "Delta arrows compare your filtered selection against the full dataset baseline."
        )

    tab_charts, tab_zone, tab_data = st.tabs(["📈 Charts", "🗺️ By Zone", "📋 Raw Data"])

    with tab_charts:
        # Annotated histogram
        median_val = filtered['Rent'].median()
        with chart_container(filtered[['Rent']]):
            fig = px.histogram(filtered, x='Rent', nbins=40, title='Rent Distribution',
                               color_discrete_sequence=['#C0392B'])
            fig.add_vline(x=median_val, line_width=2, line_color='#2C3E50',
                          annotation_text=f"Median: €{median_val:,.0f}",
                          annotation_position="top right",
                          annotation_font_color='#2C3E50')
            st.plotly_chart(fig, use_container_width=True)

        # Box plot by district
        with chart_container(filtered[['District', 'Rent']]):
            fig = px.box(filtered, x='Rent', y='District',
                         title='Rent Distribution by District',
                         color_discrete_sequence=['#C0392B'])
            fig.update_layout(yaxis={'categoryorder': 'median ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # Scatter
        with chart_container(filtered[['Sq.Mt', 'Rent', 'District']]):
            fig = px.scatter(filtered, x='Sq.Mt', y='Rent', color='District',
                             size='Rent', title='Rent vs Size', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        fig = px.imshow(
            filtered[['Rent','Sq.Mt','Bedrooms','Floor','Outer','Elevator']].corr(),
            text_auto='.2f', color_continuous_scale='RdBu_r', title='Correlation Matrix'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_zone:
        zone_summary = (
            filtered.groupby('Zone')['Rent']
            .agg(['median', 'mean', 'count'])
            .reset_index()
        )
        zone_summary.columns = ['Zone', 'Median Rent', 'Mean Rent', 'Count']
        zone_summary = zone_summary.sort_values('Median Rent', ascending=False)

        fig = px.bar(zone_summary, x='Zone', y='Median Rent',
                     title='Median Rent by Geographic Zone',
                     color='Median Rent', color_continuous_scale='Reds', text='Median Rent')
        fig.update_traces(texttemplate='€%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(filtered, x='Rent', y='Zone',
                     title='Rent Distribution by Zone',
                     color_discrete_sequence=['#C0392B'])
        fig.update_layout(yaxis={'categoryorder': 'median ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        explored = dataframe_explorer(filtered, case=False)
        st.dataframe(explored, use_container_width=True)
        st.download_button(
            "⬇️ Download filtered data as CSV",
            explored.to_csv(index=False),
            file_name="madrid_rentals_filtered.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PROPERTY SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏘️ Property Segments":

    with st.expander("How this works"):
        st.write(
            "K-Means clustering (k=5) applied to 5 property features: size (m²), bedrooms, floor, "
            "exterior exposure, and property type (special/standard). k was validated using the elbow "
            "method and silhouette analysis. Features were scaled with StandardScaler before clustering "
            "to ensure equal weighting across dimensions."
        )

    summary = df.groupby('Segment').agg(
        Properties=('Rent', 'count'),
        Median_rent=('Rent', 'median'),
        Median_SqMt=('Sq.Mt', 'median'),
        Median_Floor=('Floor', 'median'),
        Pct_Outer=('Outer', 'mean'),
        Pct_Special=('Is_Special', 'mean'),
    ).reset_index()

    seg_desc   = {name: desc for _, (name, desc) in M['segment_labels'].items()}
    seg_colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#C0392B']

    tab_overview, tab_classify = st.tabs(["📊 Overview", "🔮 Classify Property"])

    with tab_overview:
        # Segment cards
        card_cols = st.columns(len(summary))
        for i, (_, row) in enumerate(summary.iterrows()):
            color      = seg_colors[i % len(seg_colors)]
            desc_short = seg_desc.get(row['Segment'], '')[:90] + '…'
            with card_cols[i]:
                st.markdown(f"""
                <div style='border-left:4px solid {color}; background:#FAFAFA;
                            border-radius:6px; padding:10px 12px; min-height:170px;'>
                    <b style='color:{color}; font-size:0.85rem;'>{row['Segment']}</b><br>
                    <small>🏠 {int(row['Properties'])} listings</small><br>
                    <small>💶 €{int(row['Median_rent']):,} median</small><br>
                    <small>📐 {int(row['Median_SqMt'])} m²</small><br><br>
                    <small style='color:#7f8c8d;'>{desc_short}</small>
                </div>
                """, unsafe_allow_html=True)

        add_vertical_space(1)

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig = px.pie(summary, values='Properties', names='Segment',
                         title='Share of Listings per Segment',
                         color_discrete_sequence=seg_colors, hole=0.35)
            st.plotly_chart(fig, use_container_width=True)
        with col_bar:
            fig = px.bar(summary.sort_values('Median_rent'), x='Segment', y='Median_rent',
                         title='Median Rent by Segment',
                         color='Median_rent', color_continuous_scale='Reds', text='Median_rent')
            fig.update_traces(texttemplate='€%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        # Scatter
        fig = px.scatter(df, x='Sq.Mt', y='Rent', color='Segment',
                         size='Price_per_sqm', title='Rent vs Size by Segment',
                         opacity=0.7, color_discrete_sequence=seg_colors)
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart — compare segments across normalised dimensions
        radar_dims   = ['Median_rent', 'Median_SqMt', 'Median_Floor', 'Pct_Outer', 'Pct_Special']
        radar_labels = ['Median Rent', 'Median Size', 'Floor Level', 'Exterior %', 'Premium Type %']
        summary_norm = summary.copy()
        for dim in radar_dims:
            rng = summary_norm[dim].max() - summary_norm[dim].min()
            summary_norm[f'{dim}_n'] = (
                (summary_norm[dim] - summary_norm[dim].min()) / rng * 100 if rng > 0 else 50
            )
        fig_radar = go.Figure()
        for i, (_, row) in enumerate(summary_norm.iterrows()):
            vals = [row[f'{d}_n'] for d in radar_dims]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                name=row['Segment'],
                line_color=seg_colors[i % len(seg_colors)],
                opacity=0.75,
            ))
        fig_radar.update_layout(
            polar={'radialaxis': {'visible': True, 'range': [0, 100]}},
            title='Segment Profiles (each dimension normalised 0–100)',
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_classify:
        st.subheader("Which segment does this property belong to?")
        col_a, col_b = st.columns(2)
        with col_a:
            sqm      = st.number_input("Square meters", 10, 1000, 100)
            bedrooms = st.number_input("Bedrooms",       0,   10,   2)
            floor    = st.number_input("Floor",           0,   25,   1)
        with col_b:
            is_special = st.checkbox(
                "Special property (penthouse / duplex / cottage / semidetached)")
            outer = st.checkbox("Exterior facing")

        if st.button("Classify Segment", type="primary"):
            arr    = np.array([[sqm, bedrooms, floor, int(is_special), int(outer)]])
            scaled = M['cluster_scaler'].transform(arr)
            cid    = M['kmeans'].predict(scaled)[0]
            name, desc = M['segment_labels'][cid]
            st.success(f"**{name}**")
            st.info(desc)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":

    with st.expander("How this works"):
        st.write(
            "Apriori algorithm applied to binary property attributes and market segment labels. "
            "Each listing is encoded as a basket of items: exterior-facing, has elevator, "
            "special property type, central district, studio, high rent, and its market segment. "
            "The algorithm surfaces combinations that co-occur more often than chance — "
            "quantified by support (how common the combination is), confidence (how often the "
            "consequent follows the antecedent), and lift (how much more likely than random). "
            "Lift > 1 means the items attract each other; lift < 1 means they repel."
        )

    @st.cache_data
    def compute_association_rules(_df, min_support=0.03):
        binary_cols = ['Outer', 'Elevator', 'Is_Special', 'Is_Central', 'Is_Studio', 'High_Rent']
        seg_dummies = pd.get_dummies(_df['Segment'], prefix='Seg').astype(bool)
        basket = pd.concat([_df[binary_cols].astype(bool), seg_dummies], axis=1)

        freq = apriori(basket, min_support=min_support, use_colnames=True)
        if freq.empty:
            return pd.DataFrame()

        rules = association_rules(freq, metric='lift', min_threshold=1.0, num_itemsets=len(freq))
        rules['n_antecedents'] = rules['antecedents'].apply(len)
        rules['n_consequents'] = rules['consequents'].apply(len)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        rules['confidence_diff'] = (rules['confidence'] - rules['consequent support']).abs()
        lo = np.minimum(rules['consequent support'], rules['confidence'])
        hi = np.maximum(rules['consequent support'], rules['confidence'])
        rules['confidence_ratio'] = np.where(hi == 0, 0.0, 1 - lo / hi)
        return rules.sort_values('lift', ascending=False).reset_index(drop=True)

    with st.spinner("Mining association rules…"):
        all_rules = compute_association_rules(df, min_support=0.03)

    col_conf, col_lift, col_supp = st.columns(3)
    min_conf = col_conf.slider("Min Confidence", 0.0, 1.0, 0.5, step=0.05,
                               help="How often the rule is correct when the antecedent is present.")
    min_lift = col_lift.slider("Min Lift", 1.0, 12.0, 1.2, step=0.1,
                               help="How much more likely the consequent is than by chance. >1 = attraction.")
    min_supp = col_supp.slider("Min Rule Support", 0.0, 0.20, 0.03, step=0.005,
                               help="Fraction of all listings where both antecedent and consequent appear together.")

    col_cdiff, col_cratio = st.columns(2)
    min_cdiff = col_cdiff.slider("Min Confidence Difference", 0.0, 1.0, 0.0, step=0.05,
                                 help="Absolute difference between confidence and the consequent's base rate. 0 = rule matches prior, 1 = maximum divergence.")
    min_cratio = col_cratio.slider("Min Confidence Ratio", 0.0, 1.0, 0.0, step=0.05,
                                   help="1 − min(prior, confidence) / max(prior, confidence). 0 = no change from prior, 1 = maximum shift.")

    max_ant = int(all_rules['n_antecedents'].max()) if not all_rules.empty else 3
    max_con = int(all_rules['n_consequents'].max()) if not all_rules.empty else 3
    col_nant, col_ncon = st.columns(2)
    sel_ant = col_nant.number_input("Max Antecedents", min_value=1, max_value=max_ant, value=max_ant, step=1,
                                    help="Keep only rules with at most this many antecedent items.")
    sel_con = col_ncon.number_input("Max Consequents", min_value=1, max_value=max_con, value=max_con, step=1,
                                    help="Keep only rules with at most this many consequent items.")

    filtered_rules = all_rules[
        (all_rules['confidence'] >= min_conf) &
        (all_rules['lift'] >= min_lift) &
        (all_rules['support'] >= min_supp) &
        (all_rules['confidence_diff'] >= min_cdiff) &
        (all_rules['confidence_ratio'] >= min_cratio) &
        (all_rules['n_antecedents'] <= sel_ant) &
        (all_rules['n_consequents'] <= sel_con)
    ].copy()

    st.metric("Rules found", len(filtered_rules))

    if filtered_rules.empty:
        st.info("No rules match the current filters — try lowering Confidence or Lift.")
    else:
        display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift',
                        'confidence_diff', 'confidence_ratio', 'leverage']
        st.dataframe(
            filtered_rules[display_cols].round(4),
            use_container_width=True
        )

        add_vertical_space(1)

        top_n = min(15, len(filtered_rules))
        top = filtered_rules.head(top_n).copy()
        top['rule'] = top['antecedents'] + '  →  ' + top['consequents']

        fig = px.bar(
            top, x='lift', y='rule', orientation='h',
            color='confidence', color_continuous_scale='Reds',
            title=f'Top {top_n} Rules by Lift',
            labels={'lift': 'Lift', 'rule': '', 'confidence': 'Confidence'},
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=max(400, top_n * 38))
        st.plotly_chart(fig, use_container_width=True)

        add_vertical_space(1)

        # Scatter: support vs confidence, sized by lift
        fig2 = px.scatter(
            filtered_rules, x='support', y='confidence', size='lift', color='lift',
            color_continuous_scale='Reds',
            hover_data={'antecedents': True, 'consequents': True, 'lift': ':.3f'},
            title='Support vs Confidence (bubble size = Lift)',
            labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
        )
        fig2.add_hline(y=min_conf, line_dash='dot', line_color='#2C3E50')
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RENT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💶 Rent Predictor":

    with st.expander("How this works"):
        st.write(
            "OLS linear regression trained on 80% of the dataset. "
            "Multicollinearity was addressed by removing features with VIF > 10. "
            "Feature selection used RFECV (Repeated K-Fold, 5 splits × 3 repeats, MSE scoring), "
            f"retaining {len(M['selected_features'])} features: {', '.join(M['selected_features'])}. "
            "The final model is fit with statsmodels to provide p-values and 95% prediction intervals."
        )

    tab_perf, tab_predict = st.tabs(["📈 Model Performance", "🔮 Predict Rent"])

    with tab_perf:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Test", f"{M['r2_test_r']:.3f}")
        col2.metric("RMSE",    f"€{M['rmse_r']:,.0f}")
        col3.metric("MAE",     f"€{M['mae_r']:,.0f}")
        dw = sm.stats.stattools.durbin_watson(M['ols_model'].resid)
        col4.metric("Durbin-Watson", f"{dw:.3f}",
                    help="~2.0 = no autocorrelation · <1.5 or >2.5 = concern")

        # Coefficient chart
        coef_plot = M['coef_df'][M['coef_df']['Feature'] != 'const'].copy()
        coef_plot['Direction'] = coef_plot['Effect (€)'].apply(
            lambda x: 'Increases rent' if x > 0 else 'Decreases rent')
        fig = px.bar(coef_plot, x='Effect (€)', y='Feature', color='Direction',
                     orientation='h',
                     color_discrete_map={'Increases rent': '#27AE60', 'Decreases rent': '#C0392B'},
                     title='Effect of Each Feature on Rent (€)')
        st.plotly_chart(fig, use_container_width=True)

        col_scatter, col_resid = st.columns(2)
        with col_scatter:
            fig = px.scatter(x=M['y_test_r'], y=M['y_pred_r'],
                             labels={'x': 'Actual Rent (€)', 'y': 'Predicted Rent (€)'},
                             title='Actual vs Predicted Rent',
                             opacity=0.6, color_discrete_sequence=['#C0392B'])
            min_v, max_v = float(M['y_test_r'].min()), float(M['y_test_r'].max())
            fig.add_shape(type='line', x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                          line=dict(color='#2C3E50', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)

        with col_resid:
            residuals = np.array(M['y_pred_r']) - np.array(M['y_test_r'])
            fig = px.scatter(x=M['y_test_r'], y=residuals,
                             labels={'x': 'Actual Rent (€)', 'y': 'Residual (Predicted − Actual, €)'},
                             title='Residuals vs Actual Rent',
                             opacity=0.6, color_discrete_sequence=['#3498DB'])
            fig.add_hline(y=0, line_dash='dash', line_color='#2C3E50')
            st.plotly_chart(fig, use_container_width=True)

        # Q-Q plot — residual normality check
        residuals = np.array(M['y_pred_r']) - np.array(M['y_test_r'])
        (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode='markers',
            marker={'color': '#C0392B', 'opacity': 0.6, 'size': 5},
            name='Residuals',
        ))
        fig_qq.add_trace(go.Scatter(
            x=osm, y=slope * np.array(osm) + intercept,
            mode='lines', line=dict(color='#2C3E50', dash='dash'),
            name='Normal reference line',
        ))
        fig_qq.update_layout(
            title='Q-Q Plot — Residual Normality Check',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Ordered Residuals (€)',
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption(
            "Points following the dashed line indicate normally distributed residuals — "
            "a core OLS assumption. Deviation at the tails signals skewness or outliers."
        )

    with tab_predict:
        st.subheader("Predict rent for a new property")
        col_a, col_b = st.columns(2)
        with col_a:
            sqm_r      = st.number_input("Size (m²)",    15,  500, 80,  key='r_sqm',
                                         help="Total usable floor area in square metres")
            bedrooms_r = st.number_input("Bedrooms",      0,    8,  2,  key='r_bed',
                                         help="Number of bedrooms (0 = studio)")
            floor_r    = st.number_input("Floor",          0,   25,  3,  key='r_floor',
                                         help="Floor number (0 = ground floor)")
            outer_r    = st.checkbox("Exterior facing", key='r_outer',
                                     help="Does the property face an exterior street or garden?")
        with col_b:
            elevator_r   = st.checkbox("Has elevator",                   key='r_elev')
            is_special_r = st.checkbox("Penthouse / Duplex / Cottage",   key='r_spec',
                                       help="Penthouse, duplex, cottage, or semidetached")
            is_central_r = st.checkbox("Central district",               key='r_cent',
                                       help="Salamanca, Chamberí, Retiro, or Centro")
            is_studio_r  = st.checkbox("Listed as Studio",               key='r_stud',
                                       help="Property is listed with 'Estudio' in the address")

        if st.button("Predict Rent", type="primary"):
            input_dict = {
                'Sq.Mt': sqm_r, 'Bedrooms': bedrooms_r, 'Floor': floor_r,
                'Outer': int(outer_r), 'Elevator': int(elevator_r),
                'Is_Special': int(is_special_r), 'Is_Central': int(is_central_r),
                'Is_Studio': int(is_studio_r)
            }
            input_df   = pd.DataFrame([input_dict])[M['selected_features']]
            input_sm   = sm.add_constant(input_df, has_constant='add')
            prediction = M['ols_model'].predict(input_sm)[0]
            pred_sum   = M['ols_model'].get_prediction(input_sm).summary_frame(alpha=0.05)
            lo = pred_sum['obs_ci_lower'].values[0]
            hi = pred_sum['obs_ci_upper'].values[0]

            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Predicted Monthly Rent",  f"€{prediction:,.0f}")
            col_r2.metric("95% Prediction Interval", f"€{lo:,.0f} – €{hi:,.0f}")

            # Position in distribution
            percentile = (df['Rent'] < prediction).mean() * 100
            fig_dist = px.histogram(df, x='Rent', nbins=40,
                                    title='Where does this property sit in the market?',
                                    color_discrete_sequence=['#E8D5B7'])
            fig_dist.add_vline(x=prediction, line_color='#C0392B', line_width=2,
                               annotation_text=f"Your property: €{prediction:,.0f}",
                               annotation_position="top right",
                               annotation_font_color='#C0392B')
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption(
                f"This predicted rent is higher than {percentile:.0f}% of all listings in the dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HIGH RENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 High Rent Classifier":

    with st.expander("How this works"):
        st.write(
            "Logistic regression trained on 80% of the dataset with stratified splitting to preserve "
            "class balance (High Rent = Rent ≥ €1,800). VIF filtering (threshold = 10) was applied "
            "before fitting. Model performance is evaluated via AUC-ROC on both train and test sets — "
            "a near-zero gap confirms no overfitting. Odds ratios (Exp(B)) quantify each feature's "
            "multiplicative effect on the probability of High Rent."
        )

    y_prob_arr = np.array(M['y_prob_l'])
    y_test_arr = np.array(M['y_test_l'])

    tab_perf, tab_classify = st.tabs(["📈 Model Performance", "🔮 Classify Property"])

    with tab_perf:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",             f"{M['acc_l']:.2%}", help="Computed at threshold 0.50. Adjust the slider below to see threshold-specific accuracy in the Threshold Sensitivity section.")
        col2.metric("AUC Test",             f"{M['auc_test_l']:.3f}")
        col3.metric("AUC Gap (Train−Test)", f"{M['auc_train_l'] - M['auc_test_l']:.3f}")
        col4.metric("McFadden's R²",        f"{M['logit_model'].prsquared:.3f}",
                    help="Pseudo-R² for logistic regression · >0.2 is considered good fit")

        # ROC curve with AUC labels in legend
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=M['fpr_train'], y=M['tpr_train'], mode='lines',
                                 name=f"Train (AUC={M['auc_train_l']:.3f})",
                                 line_color='#3498DB'))
        fig.add_trace(go.Scatter(x=M['fpr_test'],  y=M['tpr_test'],  mode='lines',
                                 name=f"Test  (AUC={M['auc_test_l']:.3f})",
                                 line_color='#C0392B'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 line=dict(dash='dash', color='#95a5a6'), name='Baseline'))
        fig.update_layout(title='ROC Curve',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

        threshold_val = st.slider(
            "Probability cutoff for High Rent classification", 0.05, 0.95, 0.50, step=0.05,
            key='class_threshold',
            help="Probability above this value → classified as High Rent. "
                 "Lower = more sensitive (catches more High Rent but more false alarms). "
                 "This threshold applies to both the performance metrics and the classifier below."
        )
        y_thresh = (y_prob_arr >= threshold_val).astype(int)

        col_cm, col_sep = st.columns(2)

        with col_cm:
            fig = px.imshow(confusion_matrix(y_test_arr, y_thresh), text_auto=True,
                            x=['Pred Low', 'Pred High'],
                            y=['Actual Low', 'Actual High'],
                            color_continuous_scale='Reds',
                            title='Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "**TN** (top-left): correctly predicted Low Rent · "
                "**FP** (top-right): Low Rent wrongly flagged as High · "
                "**FN** (bottom-left): missed High Rent property · "
                "**TP** (bottom-right): correctly predicted High Rent"
            )

        with col_sep:
            prob_low  = M['y_prob_l'][M['y_test_l'] == 0]
            prob_high = M['y_prob_l'][M['y_test_l'] == 1]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=prob_low,  nbinsx=20, marker_color='#3498DB',
                                       opacity=0.6, name='Low Rent'))
            fig.add_trace(go.Histogram(x=prob_high, nbinsx=20, marker_color='#C0392B',
                                       opacity=0.6, name='High Rent'))
            fig.update_layout(barmode='overlay', title='Probability Separation by Class')
            fig.add_vline(x=threshold_val, line_color='#2C3E50', line_width=2,
                          annotation_text=f"Threshold: {threshold_val:.2f}",
                          annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Threshold Sensitivity")
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Accuracy",  f"{accuracy_score(y_test_arr, y_thresh):.3f}")
        tc2.metric("Precision", f"{precision_score(y_test_arr, y_thresh, zero_division=0):.3f}")
        tc3.metric("Recall",    f"{recall_score(y_test_arr, y_thresh, zero_division=0):.3f}")
        tc4.metric("F1",        f"{f1_score(y_test_arr, y_thresh, zero_division=0):.3f}")

        # Reference table with closest row highlighted
        def highlight_closest(row):
            style = ('background-color:#FDF6EC; font-weight:bold;'
                     if abs(row['Cutoff'] - threshold_val) < 0.06 else '')
            return [style] * len(row)

        st.caption("Reference table — five fixed cutoffs for quick comparison:")
        st.dataframe(M['threshold_df'].style.apply(highlight_closest, axis=1),
                     use_container_width=True)

        st.subheader("Odds Ratios")
        st.dataframe(M['odds_table'], use_container_width=True)

    with tab_classify:
        threshold_val = st.session_state.get('class_threshold', 0.50)
        st.subheader("Is this property High Rent (≥ €1,800)?")
        col_a, col_b = st.columns(2)
        with col_a:
            sqm_l      = st.number_input("Size (m²)",    15,  500, 80,  key='l_sqm',
                                         help="Total usable floor area in square metres")
            bedrooms_l = st.number_input("Bedrooms",      0,    8,  2,  key='l_bed')
            floor_l    = st.number_input("Floor",          0,   25,  3,  key='l_floor')
            outer_l    = st.checkbox("Exterior facing",            key='l_outer')
        with col_b:
            elevator_l   = st.checkbox("Has elevator",             key='l_elev')
            is_special_l = st.checkbox("Penthouse / Duplex / Cottage", key='l_spec',
                                       help="Penthouse, duplex, cottage, or semidetached")
            is_central_l = st.checkbox("Central district",         key='l_cent',
                                       help="Salamanca, Chamberí, Retiro, or Centro")
            is_studio_l  = st.checkbox("Listed as Studio",         key='l_stud')

        if st.button("Classify", type="primary"):
            input_dict = {
                'Sq.Mt': sqm_l, 'Bedrooms': bedrooms_l, 'Floor': floor_l,
                'Outer': int(outer_l), 'Elevator': int(elevator_l),
                'Is_Special': int(is_special_l), 'Is_Central': int(is_central_l),
                'Is_Studio': int(is_studio_l)
            }
            input_df    = pd.DataFrame([input_dict])[M['logit_features']]
            input_sm    = sm.add_constant(input_df, has_constant='add')
            probability = float(M['logit_model'].predict(input_sm)[0])
            label = "🔴 High Rent" if probability >= threshold_val else "🟢 Low Rent"

            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Classification",           label)
            col_r2.metric("Probability of High Rent", f"{probability:.1%}")

            gauge_chart = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': '%'},
                title={'text': "Probability of High Rent"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#C0392B' if probability >= threshold_val else '#27AE60'},
                    'steps': [{'range': [0, threshold_val * 100],   'color': '#eafaf1'},
                               {'range': [threshold_val * 100, 100], 'color': '#fdedec'}],
                    'threshold': {'line': {'color': '#2C3E50', 'width': 3}, 'value': threshold_val * 100}
                }
            ))
            st.plotly_chart(gauge_chart, use_container_width=True)
