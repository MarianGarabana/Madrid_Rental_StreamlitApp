import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_extras.add_vertical_space import add_vertical_space
from mlxtend.frequent_patterns import apriori, association_rules
from utils import apply_css, render_sidebar, chart_header, load_and_clean_data, train_all_models

st.set_page_config(page_title="Association Rules · Madrid Rental", page_icon="🏠", layout="wide")
apply_css()

df = load_and_clean_data()
with st.spinner("Training models…"):
    M = train_all_models(df)
df['Cluster'] = M['cluster_labels']
df['Segment'] = df['Cluster'].map(lambda c: M['segment_labels'][c][0])
render_sidebar(df)

st.title("Association Rules")
st.caption("Marian Garabana · MBDS Student at IE University")
st.divider()

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
        color='confidence', color_continuous_scale=['#FFB3D9', '#B72683'],
        labels={'lift': 'Lift', 'rule': '', 'confidence': 'Confidence'},
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=max(400, top_n * 38))
    chart_header("Top Rules by Lift", "Top rules ranked by lift — the ratio of observed co-occurrence to what would be expected by chance. Bar color encodes confidence: how reliably the consequent follows the antecedent.")
    st.plotly_chart(fig, use_container_width=True)

    add_vertical_space(1)

    # Scatter: support vs confidence, sized by lift
    fig2 = px.scatter(
        filtered_rules, x='support', y='confidence', size='lift', color='lift',
        color_continuous_scale=['#FFB3D9', '#B72683'],
        hover_data={'antecedents': True, 'consequents': True, 'lift': ':.3f'},
        labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
    )
    fig2.add_hline(y=min_conf, line_dash='dot', line_color='#333333')
    chart_header("Support vs Confidence", "Each rule plotted by how common it is (support) vs. how reliable it is (confidence). Bubble size and color encode lift — look for rules in the top-right with large bubbles for the strongest patterns.")
    st.plotly_chart(fig2, use_container_width=True)
