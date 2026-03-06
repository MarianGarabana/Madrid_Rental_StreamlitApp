import streamlit as st
from utils import apply_css, render_sidebar, load_and_clean_data

st.set_page_config(page_title="Madrid Rental Market", page_icon="🏠", layout="wide")
apply_css()

df = load_and_clean_data()
render_sidebar(df)

st.title("Madrid Rental Market")
st.caption("Marian Garabana · MBDS Student at IE University · IE University 2026")
st.divider()

st.markdown(
    "End-to-end machine learning dashboard analysing **~2,100 Madrid rental listings** "
    "sourced from Idealista. All models are trained live at startup and cached for instant navigation. "
    "Use the sidebar to explore each section."
)

st.markdown("### Pages")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background:#FFFFFF; border-left:4px solid #B72683; border-radius:8px; padding:16px 20px; margin-bottom:12px;">
        <div style="font-weight:700; font-size:1rem; color:#333333;">📈 Market Explorer</div>
        <div style="font-size:0.85rem; color:#555555; margin-top:4px;">
            Filter by district and rent range. Rent distributions, box plots, scatter chart, correlation heatmap, and geographic zone analysis.
        </div>
    </div>
    <div style="background:#FFFFFF; border-left:4px solid #B72683; border-radius:8px; padding:16px 20px; margin-bottom:12px;">
        <div style="font-weight:700; font-size:1rem; color:#333333;">🔵 Property Segments</div>
        <div style="font-size:0.85rem; color:#555555; margin-top:4px;">
            K-Means clustering (k=5). Segment profiles, donut chart, radar chart, and a property classifier.
        </div>
    </div>
    <div style="background:#FFFFFF; border-left:4px solid #B72683; border-radius:8px; padding:16px 20px; margin-bottom:12px;">
        <div style="font-weight:700; font-size:1rem; color:#333333;">🔗 Association Rules</div>
        <div style="font-size:0.85rem; color:#555555; margin-top:4px;">
            Apriori algorithm on binary property attributes. Interactive filters for confidence, lift, support, and rule complexity.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#FFFFFF; border-left:4px solid #B72683; border-radius:8px; padding:16px 20px; margin-bottom:12px;">
        <div style="font-weight:700; font-size:1rem; color:#333333;">🏷️ Rent Predictor</div>
        <div style="font-size:0.85rem; color:#555555; margin-top:4px;">
            OLS linear regression with VIF + RFECV. Predict monthly rent with a 95% prediction interval.
        </div>
    </div>
    <div style="background:#FFFFFF; border-left:4px solid #B72683; border-radius:8px; padding:16px 20px; margin-bottom:12px;">
        <div style="font-weight:700; font-size:1rem; color:#333333;">🔴 High Rent Classifier</div>
        <div style="font-size:0.85rem; color:#555555; margin-top:4px;">
            Logistic regression with interactive threshold. ROC curve, confusion matrix, odds ratios, and a probability gauge.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("### ML Pipeline")
st.markdown("""
| Phase | Method | Purpose |
|---|---|---|
| 0 | EDA & Cleaning | Feature engineering, outlier fixing, missing value imputation |
| 1 | K-Means (k=5) | Unsupervised property segmentation |
| 2 | Apriori | Association rule mining on property attributes |
| 3 | OLS Regression | Rent prediction with VIF filtering and RFECV |
| 4 | Logistic Regression | Binary classification — High Rent (≥ €1,800) |
""")
