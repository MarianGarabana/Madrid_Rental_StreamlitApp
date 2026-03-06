import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from utils import apply_css, render_sidebar, chart_header, load_and_clean_data, train_all_models

st.set_page_config(page_title="High Rent Classifier · Madrid Rental", page_icon="🏠", layout="wide")
apply_css()

df = load_and_clean_data()
with st.spinner("Training models…"):
    M = train_all_models(df)
render_sidebar(df)

st.title("High Rent Classifier")
st.caption("Marian Garabana · MBDS Student at IE University")
st.divider()

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
                             line_color='#6DB300'))
    fig.add_trace(go.Scatter(x=M['fpr_test'],  y=M['tpr_test'],  mode='lines',
                             name=f"Test  (AUC={M['auc_test_l']:.3f})",
                             line_color='#B72683'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(dash='dash', color='#95a5a6'), name='Baseline'))
    fig.update_layout(xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    chart_header("ROC Curve", "ROC curves for train and test sets. AUC measures the model's ability to rank High Rent above Low Rent listings. A near-zero train/test AUC gap confirms no meaningful overfitting.")
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
                        color_continuous_scale='RdPu')
        chart_header("Confusion Matrix", "TN (top-left): correctly predicted Low Rent · FP (top-right): Low Rent wrongly flagged as High · FN (bottom-left): missed High Rent property · TP (bottom-right): correctly predicted High Rent")
        st.plotly_chart(fig, use_container_width=True)

    with col_sep:
        prob_low  = M['y_prob_l'][M['y_test_l'] == 0]
        prob_high = M['y_prob_l'][M['y_test_l'] == 1]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=prob_low,  nbinsx=20, marker_color='#3498DB',
                                   opacity=0.6, name='Low Rent'))
        fig.add_trace(go.Histogram(x=prob_high, nbinsx=20, marker_color='#B72683',
                                   opacity=0.6, name='High Rent'))
        fig.update_layout(barmode='overlay')
        fig.add_vline(x=threshold_val, line_color='#333333', line_width=2,
                      annotation_text=f"Threshold: {threshold_val:.2f}",
                      annotation_position="top right")
        chart_header("Probability Separation by Class", "Predicted probability distributions for Low Rent (blue) and High Rent (red) listings. Good separation between the peaks indicates a well-calibrated model. The vertical line shows the active classification threshold.")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Threshold Sensitivity")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Accuracy",  f"{accuracy_score(y_test_arr, y_thresh):.3f}")
    tc2.metric("Precision", f"{precision_score(y_test_arr, y_thresh, zero_division=0):.3f}")
    tc3.metric("Recall",    f"{recall_score(y_test_arr, y_thresh, zero_division=0):.3f}")
    tc4.metric("F1",        f"{f1_score(y_test_arr, y_thresh, zero_division=0):.3f}")

    # Reference table with closest row highlighted
    def highlight_closest(row):
        style = ('background-color:#F4F4F4; font-weight:bold;'
                 if abs(row['Cutoff'] - threshold_val) < 0.06 else '')
        return [style] * len(row)

    chart_header("Threshold Reference Table", "Five fixed cutoffs for quick comparison. The row closest to your active threshold is highlighted.")
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
                'bar': {'color': '#B72683' if probability >= threshold_val else '#27AE60'},
                'steps': [{'range': [0, threshold_val * 100],   'color': '#eafaf1'},
                           {'range': [threshold_val * 100, 100], 'color': '#fdedec'}],
                'threshold': {'line': {'color': '#333333', 'width': 3}, 'value': threshold_val * 100}
            }
        ))
        st.plotly_chart(gauge_chart, use_container_width=True)
