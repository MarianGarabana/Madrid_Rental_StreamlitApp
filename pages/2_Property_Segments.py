import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.add_vertical_space import add_vertical_space
from utils import apply_css, render_sidebar, chart_header, load_and_clean_data, train_all_models

st.set_page_config(page_title="Property Segments · Madrid Rental", page_icon="🏠", layout="wide")
apply_css()

df = load_and_clean_data()
with st.spinner("Training models…"):
    M = train_all_models(df)
df['Cluster'] = M['cluster_labels']
df['Segment'] = df['Cluster'].map(lambda c: M['segment_labels'][c][0])
render_sidebar(df)

st.title("Property Segments")
st.caption("Marian Garabana · MBDS Student at IE University")
st.divider()

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

seg_desc        = {name: desc for _, (name, desc) in M['segment_labels'].items()}
_seg_color_list = ['#E2F46E', '#B72683', '#6DB300', '#FFB3D9', '#FF69B4']
seg_color_map   = {M['segment_labels'][i][0]: _seg_color_list[i] for i in range(5)}

tab_overview, tab_classify = st.tabs(["📊 Overview", "🔮 Classify Property"])

with tab_overview:
    # Segment expanders
    for i, (_, row) in enumerate(summary.iterrows()):
        color     = seg_color_map.get(row['Segment'], _seg_color_list[i % len(_seg_color_list)])
        full_desc = seg_desc.get(row['Segment'], '')
        st.markdown(f"""
        <details style="border:1px solid #e0e0e0; border-left:4px solid {color};
                        border-radius:6px; margin-bottom:8px; background:#FFFFFF;">
            <summary style="padding:12px 16px; font-weight:600; font-size:0.95rem;
                            color:#333333; cursor:pointer; user-select:none;">
                {row['Segment']}
            </summary>
            <div style="padding:4px 16px 14px 16px; border-top:1px solid #f0f0f0;">
                <span style="margin-right:16px;">🏠 <b>{int(row['Properties'])}</b> listings</span>
                <span style="margin-right:16px;">💶 <b>€{int(row['Median_rent']):,}</b> median rent</span>
                <span>📐 <b>{int(row['Median_SqMt'])} m²</b> median size</span>
                <p style="margin-top:10px; color:#555555; margin-bottom:0;">{full_desc}</p>
            </div>
        </details>
        """, unsafe_allow_html=True)

    add_vertical_space(1)

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig = px.pie(summary, values='Properties', names='Segment',
                     color='Segment', color_discrete_map=seg_color_map, hole=0.35)
        chart_header("Share of Listings per Segment", "Proportion of all listings assigned to each K-Means cluster. Larger slices represent the dominant rental stock in Madrid.")
        st.plotly_chart(fig, use_container_width=True)
    with col_bar:
        fig = px.bar(summary.sort_values('Median_rent', ascending=False), x='Segment', y='Median_rent',
                     color='Segment', color_discrete_map=seg_color_map, text='Median_rent',
                     labels={'Segment': '', 'Median_rent': ''})
        fig.update_traces(texttemplate='€%{text:,.0f}', textposition='outside')
        fig.update_layout(xaxis_title='', yaxis_title='', showlegend=False)
        chart_header("Median Rent by Segment", "Median monthly rent for each segment. Highlights the price gap between budget interiors and premium/estate properties.")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter
    fig = px.scatter(df, x='Sq.Mt', y='Rent', color='Segment',
                     size='Price_per_sqm',
                     opacity=0.7, color_discrete_map=seg_color_map)
    chart_header("Rent vs Size by Segment", "Each listing plotted by size (m²) vs. rent, colored by segment. Bubble size reflects price per m² — useful for spotting value vs. premium positioning within each cluster.")
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
            line_color=seg_color_map.get(row['Segment'], _seg_color_list[i % len(_seg_color_list)]),
            opacity=0.75,
        ))
    fig_radar.update_layout(
        polar={'radialaxis': {'visible': True, 'range': [0, 100]}},
        height=500,
    )
    chart_header("Segment Profiles", "Radar chart comparing all five segments across key dimensions, each normalised to 0–100. Larger area = more extreme profile; useful for spotting how clusters differ holistically.")
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
