import streamlit as st
import plotly.express as px
from streamlit_extras.chart_container import chart_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
from utils import apply_css, render_sidebar, chart_header, load_and_clean_data

st.set_page_config(page_title="Market Explorer · Madrid Rental", page_icon="🏠", layout="wide")
apply_css()

df = load_and_clean_data()
render_sidebar(df)

st.title("Market Explorer")
st.caption("Marian Garabana · MBDS Student at IE University")
st.divider()

rent_min, rent_max = int(df['Rent'].min()), int(df['Rent'].max())

# ── Filters above KPIs ──────────────────────────────────────────────────
f_col1, f_col2 = st.columns([2, 1])
with f_col1:
    selected_rent = st.slider("Rent Range (€)", rent_min, rent_max, (rent_min, rent_max))
with f_col2:
    all_districts = sorted(df['District'].unique().tolist())
    district_options = ["All"] + all_districts
    selected_district_opts = st.multiselect(
        "Districts", district_options, default=["All"],
        placeholder="Choose districts…"
    )
    selected_districts = (
        all_districts
        if ("All" in selected_district_opts or len(selected_district_opts) == 0)
        else selected_district_opts
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
        "Use the rent slider and district dropdown above to filter the dataset. "
        "Delta arrows compare your filtered selection against the full dataset baseline."
    )

tab_charts, tab_zone, tab_data = st.tabs(["📈 Charts", "🗺️ By Zone", "📋 Raw Data"])

with tab_charts:
    # Annotated histogram
    median_val = filtered['Rent'].median()
    with chart_container(filtered[['Rent']]):
        fig = px.histogram(filtered, x='Rent', nbins=40,
                           color_discrete_sequence=['#B72683'],
                           labels={'Rent': '', 'count': ''})
        fig.update_layout(xaxis_title='', yaxis_title='')
        fig.add_vline(x=median_val, line_width=2, line_color='#333333',
                      annotation_text=f"Median: €{median_val:,.0f}",
                      annotation_position="top right",
                      annotation_font_color='#333333')
        chart_header("Rent Distribution", "Distribution of monthly rent prices across the filtered selection. The vertical line marks the median rent.")
        st.plotly_chart(fig, use_container_width=True)

    # Box plot by district
    with chart_container(filtered[['District', 'Rent']]):
        fig = px.box(filtered, x='Rent', y='District',
                     color_discrete_sequence=['#B72683'])
        fig.update_layout(yaxis={'categoryorder': 'median ascending'}, xaxis_title='', yaxis_title='')
        chart_header("Rent Distribution by District", "Box plots showing the spread, median, and outliers of rent by district, ordered from lowest to highest median.")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter
    with chart_container(filtered[['Sq.Mt', 'Rent', 'District']]):
        fig = px.scatter(filtered, x='Sq.Mt', y='Rent', color='District',
                         size='Rent', opacity=0.7)
        chart_header("Rent vs Size", "Scatter plot of property size (m²) against monthly rent, colored by district. Bubble size scales with rent value.")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    chart_header("Correlation Matrix", "Pairwise Pearson correlations between numeric features. Purple indicates strong positive correlation; lime green indicates inverse correlation.")
    fig = px.imshow(
        filtered[['Rent', 'Sq.Mt', 'Bedrooms', 'Floor', 'Outer', 'Elevator']].corr(),
        text_auto='.2f', color_continuous_scale=[[0, '#E2F46E'], [0.5, '#FFFFFF'], [1, '#B72683']]
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

    chart_header("Median Rent by Geographic Zone", "Aggregated median rent grouped by broad geographic zone. Bar color reflects the rent level — darker red means higher median rent.")
    fig = px.bar(zone_summary, x='Zone', y='Median Rent',
                 color='Median Rent', color_continuous_scale='RdPu', text='Median Rent',
                 labels={'Zone': '', 'Median Rent': ''})
    fig.update_traces(texttemplate='€%{text:,.0f}', textposition='outside')
    fig.update_layout(xaxis_title='', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)

    chart_header("Rent Distribution by Zone", "Box plots comparing the full rent spread across geographic zones, ordered by median rent. Whiskers extend to 1.5× IQR; dots are outliers.")
    fig = px.box(filtered, x='Rent', y='Zone',
                 color_discrete_sequence=['#B72683'])
    fig.update_layout(yaxis={'categoryorder': 'median ascending'}, xaxis_title='', yaxis_title='')
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
