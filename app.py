import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.ticker as mticker
from catboost import CatBoostRegressor, Pool

from io import BytesIO
import base64

# Set Streamlit page config
st.set_page_config(page_title="Marketing Dashboard & Predictor", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_marketing_campaign_dataset.csv")

df = load_data()

# ----------------------------------
# Custom Color Palettes
# ----------------------------------
# def get_custom_palette(cmap_name, n_colors=6):
#     cmap = cm.get_cmap(cmap_name)
#     return [mcolors.to_hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]

color_options = {
    "Sky Blue": sns.color_palette("Blues", 6).as_hex(),
    "Green Forest": sns.color_palette("Greens", 6).as_hex(),
    "Sunset Orange": sns.color_palette("Oranges", 6).as_hex(),
    "Deep Purple": sns.color_palette("Purples", 6).as_hex(),
    "Vibrant Magma": sns.color_palette("magma", 6).as_hex(),
    "Cool Viridis": sns.color_palette("viridis", 6).as_hex(),
    "Classic Seaborn": sns.color_palette("deep", 6).as_hex(),
}


# ----------------------------------
# Sidebar Controls
# ----------------------------------
company_selection = st.sidebar.selectbox(
    "Select Company to Analyze",
    ["All Companies",  "Alpha Innovations", "DataTech Solutions", "Innovate Industries", "NexGen Systems", "TechCorp"]
)

page = st.sidebar.radio("Select View", ["Marketing Dashboard", "Conversion Rate Predictor"])

# Apply company filter
if company_selection != "All Companies":
    df = df[df["Company"] == company_selection]


# ----------------------------
# DASHBOARD FUNCTION
# ----------------------------
def show_dashboard(df, company):
    st.title("Marketing Campaign Dashboard")
    st.markdown(f"### **{company}**")

    # Color Theme
    with st.container():
        st.markdown("###### Select a Color Theme")
        selected_color_label = st.selectbox("Choose Visualization Theme", list(color_options.keys()))
        selected_palette = color_options[selected_color_label]

    # Campaign Type Filter
    with st.container():
        st.markdown("###### Filter by Campaign Type")
        campaign_types = df["Campaign_Type"].unique().tolist()
        selected_campaign_types = st.multiselect(
            "Select Campaign Types to View",
            options=sorted(campaign_types),
            default=sorted(campaign_types)
        )
        df = df[df["Campaign_Type"].isin(selected_campaign_types)]

    # Metrics Calculation
    total_campaigns = len(df)
    total_cost = df["Acquisition_Cost"].sum()
    total_conversion = df["Conversion"].sum()
    total_clicks = df["Clicks"].sum()

    avg_roi = df["ROI"].mean()
    avg_cvr = df["Conversion_Rate"].mean() * 100  # percent
    cost_per_conversion = total_cost / total_conversion if total_conversion > 0 else 0
    # cost_per_click = total_cost / total_clicks if total_clicks > 0 else 0


    # ----------------------------
    # 3-Column Layout
    # ----------------------------

    def human_format(num, precision=1):
        #Format large numbers in human-readable compact form, e.g., 1.2K, 3.5M
        for unit in ['', 'K', 'M', 'B', 'T']:
            if abs(num) < 1000:
                return f"{num:.{precision}f}{unit}"
            num /= 1000
        return f"{num:.{precision}f}T"

    col1, col2, col3 = st.columns([0.5, 1.5, 1])

    
    # Column 1: Metrics
    with col1:
        st.markdown("#### Key Metrics")
        st.metric("Total Campaigns", f"{human_format(total_campaigns, 0)}")
        st.metric("Ad Cost", f"${human_format(total_cost)}")
        st.metric("Avg ROI", f"{avg_roi:.2f}%")
        st.metric("Avg CVR", f"{avg_cvr:.2f}%")
        st.metric("Impressions", f"{human_format(df['Impressions'].sum(), 0)}")
        st.metric("Clicks", f"{human_format(total_clicks, 0)}")
        st.metric("Conversions", f"{human_format(total_conversion, 0)}")
        st.metric("Cost per Conversion", f"${cost_per_conversion:,.2f}")
        # st.metric("Cost per Click", f"${cost_per_click:,.2f}")

    
    # Column 2: Grouped Bar (Impressions & Clicks) + Conversion Line (Dual Y-Axis)
    with col2:
        st.markdown("#### Channel Performance Overview")
        
        # Aggregate data
        agg = df.groupby("Channel_Used")[["Impressions", "Clicks", "Conversion"]].sum().reset_index()

        # Create figure
        fig = go.Figure()

        # Impressions Bar
        fig.add_trace(go.Bar(
            x=agg["Channel_Used"],
            y=agg["Impressions"],
            name="Impressions",
            marker_color=selected_palette[1],
            yaxis="y1",
            hovertemplate='Channel: %{x}<br>Impressions: %{y:,}<extra></extra>'
        ))

        # Clicks Bar
        fig.add_trace(go.Bar(
            x=agg["Channel_Used"],
            y=agg["Clicks"],
            name="Clicks",
            marker_color=selected_palette[2],
            yaxis="y1",
            hovertemplate='Channel: %{x}<br>Clicks: %{y:,}<extra></extra>'
        ))

        # Conversions Line (right Y-axis)
        fig.add_trace(go.Scatter(
            x=agg["Channel_Used"],
            y=agg["Conversion"],
            name="Conversions",
            mode="lines+markers",
            marker=dict(color=selected_palette[4]),
            yaxis="y2",
            line=dict(width=2),
            hovertemplate='Channel: %{x}<br>Conversions: %{y:.4f}<extra></extra>'
        ))

        # Layout and axes (corrected)
        fig.update_layout(
            barmode='group',
            xaxis=dict(title="Channel Used"),
            yaxis=dict(
                title=dict(text="Impressions / Clicks", font=dict(color="black")),
                tickfont=dict(color="black")
            ),
            yaxis2=dict(
                title=dict(text="Conversions", font=dict(color="black")),
                tickfont=dict(color="black"),
                overlaying='y',
                side='right'
            ),
            legend=dict(x=1, y=1),
            margin=dict(l=40, r=40, t=20, b=40),
            height=450
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)
           
        # Aggregate data
        agg_bubble = df.groupby("Channel_Used").agg({
            "ROI": "mean",
            "Conversion_Rate": "mean"
        }).reset_index()

        # Convert Conversion Rate to percentage
        agg_bubble["Conversion_Rate"] = agg_bubble["Conversion_Rate"] * 100

        # Prepare color map using selected_palette
        unique_channels = agg_bubble["Channel_Used"].unique()
        color_map = {
            channel: selected_palette[i % len(selected_palette)]
            for i, channel in enumerate(unique_channels)
        }

        # Plot with custom colors (no bubble size)
        fig_bubble = px.scatter(
            agg_bubble,
            x="Conversion_Rate",
            y="ROI",
            color="Channel_Used",
            color_discrete_map=color_map,
            text="Channel_Used",
            hover_data={
                "ROI": ":.2f",
                "Conversion_Rate": ":.2f"
            },
            labels={
                "ROI": "Avg ROI (%)",
                "Conversion_Rate": "Avg CVR (%)"
            },
            height=450
        )

        fig_bubble.update_traces(
            textposition='top center',
            marker=dict(size=15, line=dict(width=0, color='black'))  # fixed bubble size
        )

        fig_bubble.update_layout(
            margin=dict(l=40, r=40, t=10, b=200),
            showlegend=False 
        )

        st.plotly_chart(fig_bubble, use_container_width=True)

    
    # Column 3: Engagement & Grouped Bars
    with col3:
        st.markdown("#### Audience Segment Performance")

        # Average Conversion Rate by Location
        location_df = (
            df.groupby("Location")["Conversion_Rate"]
            .mean()
            .reset_index()
            .sort_values(by="Conversion_Rate", ascending=False)
            .reset_index(drop=True)
        )

        # Multiply by 100 for percentage format
        location_df["Avg CVR (%)"] = (location_df["Conversion_Rate"] * 100).round(2)
        location_df = location_df.drop(columns=["Conversion_Rate"])

        # Display DataFrame
        st.dataframe(location_df, height=140)

        # Prepare data for gender-age conversion rate chart
        df_gender_age = (
            df.groupby(["Gender", "Age_Group"])["Conversion_Rate"]
            .mean()
            .reset_index()
        )

        # Convert to percentage
        df_gender_age["Conversion_Rate (%)"] = df_gender_age["Conversion_Rate"] * 100

        # Dynamically assign colors based on gender values and selected_palette
        unique_genders = df_gender_age["Gender"].unique()
        gender_color_map = {
            gender: selected_palette[i % len(selected_palette)]
            for i, gender in enumerate(unique_genders)
        }

        # Create Sunburst chart
        fig_sunburst = px.sunburst(
            df_gender_age,
            path=["Gender", "Age_Group"],
            values="Conversion_Rate",
            color="Gender",
            color_discrete_map=gender_color_map,
            hover_data={"Conversion_Rate": ":.2%", "Conversion_Rate (%)": False}
        )

        fig_sunburst.update_layout(
            margin=dict(t=20, l=40, r=40, b=20),
            height=285,  # adjust to make it smaller vertically
            width=285    # adjust to make it smaller horizontally
        )

        st.plotly_chart(fig_sunburst, use_container_width=True)

        # Prepare data
        df_segment = (
            df.groupby("Customer_Segment")["Conversion_Rate"]
            .mean()
            .reset_index()
        )
        df_segment["Conversion_Rate (%)"] = df_segment["Conversion_Rate"] * 100

        # Dynamically create color map from selected_palette
        unique_segments = df_segment["Customer_Segment"].unique()
        segment_color_map = {
            segment: selected_palette[i % len(selected_palette)]
            for i, segment in enumerate(unique_segments)
        }

        # Create Treemap
        fig_segment = px.treemap(
            df_segment,
            path=["Customer_Segment"],
            values="Conversion_Rate",
            color="Customer_Segment",
            color_discrete_map=segment_color_map,
            hover_data={"Conversion_Rate": ":.2%", "Conversion_Rate (%)": False}
        )

        # Layout adjustments
        fig_segment.update_layout(
            margin=dict(t=0, l=40, r=40, b=20),
            height=330
        )

        # Show in Streamlit
        st.plotly_chart(fig_segment, use_container_width=True)


# ----------------------------
# PREDICTION INTERFACE FUNCTION
# ----------------------------
def show_prediction_interface():
    st.title("Conversion Rate Predictor")

    company = company_selection if company_selection != "All Companies" else st.selectbox(
        "Select Company to Predict",
        ["Innovate Industries", "NexGen Systems", "Alpha Innovations", "DataTech Solutions", "TechCorp"]
    )

    # Load model & metadata
    model_path = f"catboost_{company.replace(' ', '_').lower()}.cbm"
    meta_path = f"meta_{company.replace(' ', '_').lower()}.pkl"

    model = CatBoostRegressor()
    model.load_model(model_path)

    meta = joblib.load(meta_path)
    categorical_cols = meta['categorical_cols']
    column_order = meta['column_order']

    st.subheader("Enter Campaign Details")

    # --- Section 1: General campaign info (no impressions/clicks yet) ---
    user_input = {
        "Campaign_Type": st.selectbox("Campaign Type", sorted(["Display", "Email", "Social Media", "Search", "Influencer"])),
        "Channel_Used": st.selectbox("Channel Used", sorted(["Google Ads", "YouTube", "Facebook", "Instagram", "Email", "Website"])),
        "Customer_Segment": st.selectbox("Customer Segment", sorted(["Tech Enthusiasts", "Foodies", "Health & Wellness", "Outdoor Adventurers", "Fashionistas"])),
        "Gender": st.selectbox("Target Gender", sorted(["Male", "Female", "All"])),
        "Age_Group": st.selectbox("Target Age Group", sorted(["18-24", "25-34", "All Ages", "35-44"])),
        "Location": st.selectbox("Target Location", sorted(["Miami", "New York", "Chicago", "Los Angeles", "Houston"])),
        "Language": st.selectbox("Language", sorted(["English", "Spanish", "Mandarin", "German", "French"])),
        "Duration": st.number_input("Campaign Duration (days)", 0, 120, step=1),
        "ROI": st.number_input("ROI (Return on Investment) (%)", 0.0, 10.0, step=0.01),
        "Acquisition_Cost": st.number_input("Acquisition Cost (in USD)", 0, 20000, step=100),
        "Engagement_Score": st.number_input("Engagement Score (0 to 10)", 0, 10, step=1),
    }

    st.markdown("---")

    impressions = st.number_input("Number of Impressions", 1, 100000, step=100)
    clicks = st.number_input("Number of Clicks", 0, 10000, step=100)

    # Calculate CTR
    ctr = round(clicks / impressions, 4) if impressions > 0 else 0.0
    st.info(f"**Estimated CTR (Click-Through Rate):** {ctr:.4f}")

    # Add to user input dictionary
    user_input["Clicks"] = clicks
    user_input["Impressions"] = impressions
    user_input["CTR"] = ctr

    input_df = pd.DataFrame([user_input])
    input_df = input_df[column_order]
    cat_features_indices = [input_df.columns.get_loc(col) for col in categorical_cols]

    if st.button("Run Prediction"):
        # Predict
        user_pool = Pool(input_df, cat_features=cat_features_indices)
        prediction = model.predict(user_pool)[0]
        st.success(f"**Predicted Conversion Rate:** {prediction * 100:.2f}%")

        # SHAP Analysis
        st.subheader("SHAP Feature Contribution (Waterfall)")
        st.caption("""
        This SHAP waterfall chart shows how each input feature contributed to the final predicted conversion rate:

        - Features are ordered from **most to least impactful** (left to right).
        - **Positive bars (green)** push the prediction **upwards** (increase conversion rate).
        - **Negative bars (red)** push the prediction **downwards** (decrease conversion rate).
        - The final bar labeled **"Predicted CVR" (blue)** shows the total result after all feature contributions are applied to the model’s base value (average prediction).
        """)
        # SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(input_df)

        # Get top features and their SHAP values
        shap_values = shap_explanation.values[0]
        expected_value = shap_explanation.base_values[0]
        predicted_value = expected_value + shap_values.sum()

        # Combine feature names, user inputs, and SHAP values
        feature_names = shap_explanation.feature_names
        input_values = input_df.iloc[0].values
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "input_value": input_values,
            "shap_value": shap_values
        })

        # Sort by absolute SHAP value
        shap_df["abs_val"] = shap_df["shap_value"].abs()
        shap_df_sorted = shap_df.sort_values("abs_val", ascending=False).head(10)

        # Plotly Waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=["relative"] * len(shap_df_sorted) + ["total"],
            x=[f"{row['feature']} = {row['input_value']}" for _, row in shap_df_sorted.iterrows()] + ["Predicted CVR"],
            text=[f"{row['shap_value']:+.6f}" for _, row in shap_df_sorted.iterrows()] + [f"{predicted_value:.6f}"],
            textposition="outside",
            y=list(shap_df_sorted["shap_value"]) + [predicted_value - expected_value],
            connector={"line": {"color": "gray"}}
        ))

        fig.update_layout(
            title="SHAP Feature Contribution (Waterfall Plot)",
            yaxis_title="SHAP Value (Impact on Prediction)",
            showlegend=False,
            height=500
        )

        # Display
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# MAIN VIEW SWITCHER
# ----------------------------
if page == "Marketing Dashboard":
    show_dashboard(df, company_selection)
else:
    show_prediction_interface()


    
# PLEASE type "python -m streamlit run app.py" in terminal to run this file
