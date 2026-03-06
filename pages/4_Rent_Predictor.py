import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from utils import apply_css, render_sidebar, chart_header, load_and_clean_data, train_all_models

st.set_page_config(page_title="Rent Predictor · Madrid Rental", page_icon="🏠", layout="wide")
apply_css()

df = load_and_clean_data()
with st.spinner("Training models…"):
    M = train_all_models(df)
render_sidebar(df)

st.title("Rent Predictor")
st.caption("Marian Garabana · MBDS Student at IE University")
st.divider()

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
                 color_discrete_map={'Increases rent': '#E2F46E', 'Decreases rent': '#B72683'})
    chart_header("Feature Effects on Rent", "OLS regression coefficients expressed in euros. Each bar shows the estimated change in monthly rent for a one-unit increase in that feature, holding all others constant.")
    st.plotly_chart(fig, use_container_width=True)

    col_scatter, col_resid = st.columns(2)
    with col_scatter:
        fig = px.scatter(x=M['y_test_r'], y=M['y_pred_r'],
                         labels={'x': 'Actual Rent (€)', 'y': 'Predicted Rent (€)'},
                         opacity=0.6, color_discrete_sequence=['#B72683'])
        min_v, max_v = float(M['y_test_r'].min()), float(M['y_test_r'].max())
        fig.add_shape(type='line', x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                      line=dict(color='#333333', dash='dash'))
        chart_header("Actual vs Predicted Rent", "Each point is a test-set property. Points near the dashed diagonal indicate accurate predictions; vertical spread reveals the model's error range.")
        st.plotly_chart(fig, use_container_width=True)

    with col_resid:
        residuals = np.array(M['y_pred_r']) - np.array(M['y_test_r'])
        fig = px.scatter(x=M['y_test_r'], y=residuals,
                         labels={'x': 'Actual Rent (€)', 'y': 'Residual (Predicted − Actual, €)'},
                         opacity=0.6, color_discrete_sequence=['#E2F46E'])
        fig.add_hline(y=0, line_dash='dash', line_color='#333333')
        chart_header("Residuals vs Actual Rent", "Residuals should scatter randomly around zero (dashed line) with no visible pattern. A fan shape would signal heteroscedasticity; a curve would suggest a non-linear relationship.")
        st.plotly_chart(fig, use_container_width=True)

    # Q-Q plot — residual normality check
    residuals = np.array(M['y_pred_r']) - np.array(M['y_test_r'])
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=osm, y=osr, mode='markers',
        marker={'color': '#B72683', 'opacity': 0.6, 'size': 5},
        name='Residuals',
    ))
    fig_qq.add_trace(go.Scatter(
        x=osm, y=slope * np.array(osm) + intercept,
        mode='lines', line=dict(color='#333333', dash='dash'),
        name='Normal reference line',
    ))
    fig_qq.update_layout(
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Ordered Residuals (€)',
    )
    chart_header("Q-Q Plot — Residual Normality Check", "Points following the dashed line indicate normally distributed residuals — a core OLS assumption. Deviation at the tails signals skewness or outliers.")
    st.plotly_chart(fig_qq, use_container_width=True)

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
                                color_discrete_sequence=['#D9D9D9'])
        fig_dist.add_vline(x=prediction, line_color='#B72683', line_width=2,
                           annotation_text=f"Your property: €{prediction:,.0f}",
                           annotation_position="top right",
                           annotation_font_color='#B72683')
        chart_header("Market Position", f"This predicted rent is higher than {percentile:.0f}% of all listings in the dataset.")
        st.plotly_chart(fig_dist, use_container_width=True)
