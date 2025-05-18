import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Page config
st.set_page_config(
    page_title="📱 Time-Wasters Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* General typography */
    .main {
        background-color: #fffefc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1, h2, h3 {
        color: #ff5a36;
        font-weight: 1000;
    }

    p {
        font-weight: 1000 !important;
        
        color: #ff5a36;
       font-size:28px;
        padding: 10px;
    }

    /* Expander headers */
    details > summary {
        color: black !important;
        font-weight: 900 !important;
        font-size: 18px !important;
    }

    details summary::marker {
        color: black !important;
    }

    /* Card styles */
    .card {
        background: white;
        padding: 20px;
        margin: 10px 0;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(255,90,54,0.15);
        transition: box-shadow 0.3s ease-in-out;
    }

    .card:hover {
        box-shadow: 0 12px 20px rgba(255,90,54,0.3);
    }

    /* Sidebar base */
[data-testid="stSidebar"] {
    background-color: #ff5a36         !important; /* Orange background */
    padding: 20px 15px;
    border-radius: 15px;
    color: black !important;
    font-weight: 1500;
    font-size: 1rem;
}

/* Sidebar headings (black text) */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] p {
    color: black !important;
    font-weight: 1400 !important;
}

/* Sidebar expander labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .st-expanderHeader {
    color: black !important;
    font-weight: 1000 !important;
}

/* Sidebar widget text (black on orange) */
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stMultiSelect > div,
[data-testid="stSidebar"] .stSlider > div,
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stTextInput {
    color: black !important;
}

/* Input fields background (white) */
.stSelectbox div[data-baseweb], 
.stMultiSelect div[data-baseweb],
.stSlider,
input[type="text"],
input[type="number"],
textarea {
    background-color:black !important;
    color: white !important;
    font-color:white;
    border-radius: 12px;
}


    /* Buttons */
    div.stButton > button {
        background-color: #ffffff !important;
        color: #ff5a36 !important;
        font-weight: 900 !important;
        border-radius: 12px !important;
        padding: 10px 22px !important;
        border: 2px solid #ff5a36 !important;
        font-size: 20px !important;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #ff5a36 !important;
        color: white !important;
        border: 2px solid #cc4628 !important;
    }
    </style>
""", unsafe_allow_html=True)




# Header banner
st.markdown("""
    <div style="background-color:#ff5a36; padding:15px; border-radius:15px; text-align:center; color:white; font-size:28px; font-weight:700; margin-bottom:25px;">
        📱 Social Media Engagement Insights Dashboard
    </div>
""", unsafe_allow_html=True)

# Load data
df = pd.read_csv('data/Time-Wasters on Social Media.csv')
df.columns = df.columns.str.strip()

# Sidebar filters with grouping
with st.sidebar:
    st.header("🎛️ Filters")
    with st.expander("User Info Filters", expanded=True):
        platform = st.selectbox("Platform", df["Platform"].unique())
        age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 60))
        genders = st.multiselect("Gender", options=df["Gender"].unique(), default=list(df["Gender"].unique()))
        locations = st.multiselect("Location", options=df["Location"].unique(), default=list(df["Location"].unique()))
        professions = st.multiselect("Profession", df["Profession"].unique(), default=list(df["Profession"].unique()))
    with st.expander("Device & Usage Filters", expanded=False):
        devices = st.multiselect("Device Type", df["DeviceType"].unique(), default=list(df["DeviceType"].unique()))
        activities = st.multiselect("Current Activity", df["CurrentActivity"].unique(), default=list(df["CurrentActivity"].unique()))
        addiction_range = st.slider("Addiction Level", 0, 10, (0, 10))
        connection_type = st.selectbox("Connection Type", df["ConnectionType"].unique())

# Filtering data
filtered_df = df[
    (df["Platform"] == platform) &
    (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
    (df["Gender"].isin(genders)) &
    (df["Location"].isin(locations)) &
    (df["DeviceType"].isin(devices)) &
    (df["CurrentActivity"].isin(activities)) &
    (df["Profession"].isin(professions)) &
    (df["Addiction Level"] >= addiction_range[0]) & (df["Addiction Level"] <= addiction_range[1]) &
    (df["ConnectionType"] == connection_type)
]

# Tabs for clean navigation
tabs = st.tabs(["📊 Summary", "📈 Visuals", "🔍 Advanced Insights", "🤖 Predictions & Tips"])

# Summary tab with cards and columns
with tabs[0]:
    st.subheader("📊 Summary Statistics")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(filtered_df.describe())
    st.markdown('</div>', unsafe_allow_html=True)

    # KPIs in columns with cards
    col1, col2, col3 = st.columns(3)
    avg_engagement = filtered_df["Engagement"].mean()
    avg_time_spent = filtered_df["Time Spent On Video"].mean()
    avg_productivity_loss = filtered_df["ProductivityLoss"].mean()

    with col1:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.metric(label="🔥 Avg. Engagement", value=f"{avg_engagement:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.metric(label="⏱️ Avg. Time on Video (mins)", value=f"{avg_time_spent:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.metric(label="📉 Avg. Productivity Loss", value=f"{avg_productivity_loss:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Engagement Distribution plot with container and spacing
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Engagement Distribution")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df["Engagement"], kde=True, ax=ax4, color='#ff5a36')
    ax4.set_xlabel("Engagement Level")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Histogram of Engagement")
    st.pyplot(fig4)
    st.markdown('</div>', unsafe_allow_html=True)

# Visuals tab - organized with columns and expanded cards
with tabs[1]:
    st.subheader("📈 Visualizations")

    # Engagement vs Age & Time Spent by Video Category side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Engagement vs Age")
        fig = px.scatter(
            filtered_df, x="Age", y="Engagement", color="Gender",
            labels={"Age": "Age", "Engagement": "Engagement Level"},
            color_discrete_sequence=px.colors.sequential.Oranges)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Time Spent on Video by Category")
        fig2 = px.bar(
            filtered_df, x="Video Category", y="Time Spent On Video", color="Gender",
            color_discrete_sequence=px.colors.sequential.PuRd)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Location engagement and device pie
    col3, col4 = st.columns([3,1])

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Engagement by Location")
        fig3 = px.bar(
            filtered_df, x="Location", y="Engagement", color="Platform",
            color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Device Type Distribution")
        device_counts = filtered_df['DeviceType'].value_counts().reset_index()
        device_counts.columns = ['DeviceType', 'Count']
        fig9 = px.pie(device_counts, names='DeviceType', values='Count',
                      color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # New Map Visualization
    st.subheader("🗺️ Engagement by Country Map")
    engagement_by_country = filtered_df.groupby("Location")["Engagement"].mean().reset_index()

    fig_map = px.choropleth(
        engagement_by_country,
        locations="Location",
        locationmode="country names",
        color="Engagement",
        color_continuous_scale=px.colors.sequential.Plasma,
        title="Average Engagement by Country",
        labels={"Engagement": "Avg Engagement"},
        hover_name="Location"
    )
    fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)


# Advanced Insights tab with neat layout
with tabs[2]:
    st.subheader("🔍 Advanced Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Addiction vs Productivity Loss")
        fig5 = px.scatter(
            filtered_df, x="Addiction Level", y="ProductivityLoss", size="Engagement", color="Gender",
            color_discrete_sequence=px.colors.sequential.Magenta)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Time Spent on Video by Profession")
        fig6 = px.bar(
            filtered_df, x="Profession", y="Time Spent On Video", color="Gender",
            color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Self Control vs Engagement")
        fig7 = px.scatter(
            filtered_df, x="Self Control", y="Engagement", color="Platform",
            hover_data=["Age", "DeviceType"],
            color_discrete_sequence=px.colors.sequential.Agsunset)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Device Type vs Time Spent")
        fig8 = px.violin(
            filtered_df, y="Time Spent On Video", x="DeviceType", box=True, color="DeviceType",
            color_discrete_sequence=px.colors.sequential.Magma)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # User Clustering with clear section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("User Segmentation Clusters")

    cluster_features = filtered_df[["Engagement", "Addiction Level", "Time Spent On Video"]].dropna()
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(cluster_scaled)
    cluster_features["Cluster"] = clusters
    risk_map = {0: "Moderate Risk", 1: "High Risk", 2: "Low Risk"}
    cluster_features["Risk Level"] = cluster_features["Cluster"].map(risk_map)

    fig_cluster = px.scatter(
        cluster_features, x="Engagement", y="Addiction Level", color="Risk Level",
        color_discrete_map={"High Risk":"#FF4B4B", "Moderate Risk":"#FFA500", "Low Risk":"#28A745"},
        title="User Clusters by Engagement & Addiction Level"
    )
    st.plotly_chart(fig_cluster)
    st.markdown('</div>', unsafe_allow_html=True)

# Predictions and Recommendations tab with card style form
with tabs[3]:
    st.subheader("🤖 Predictive Insights & Personalized Recommendations")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Prepare model data
    model_features = ["Age", "Engagement", "Time Spent On Video", "Self Control"]
    target_addiction = "Addiction Level"
    target_productivity = "ProductivityLoss"

    model_df = df[model_features + [target_addiction, target_productivity]].dropna()

    X = model_df[model_features]
    y_addiction = model_df[target_addiction]
    y_productivity = model_df[target_productivity]

    model_addiction = LinearRegression().fit(X, y_addiction)
    model_productivity = LinearRegression().fit(X, y_productivity)

    # User input form
    with st.form("prediction_form"):
        input_age = st.number_input("Age", min_value=10, max_value=100, value=25)
        input_engagement = st.slider("Engagement Level", 0, 100, 50)
        input_time_spent = st.slider("Time Spent On Video (minutes)", 0, 300, 30)
        input_self_control = st.slider("Self Control (0-10)", 0, 10, 5)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame({
                "Age": [input_age],
                "Engagement": [input_engagement],
                "Time Spent On Video": [input_time_spent],
                "Self Control": [input_self_control]
            })

            pred_addiction = model_addiction.predict(input_df)[0]
            pred_productivity = model_productivity.predict(input_df)[0]

            st.markdown(f"### Predicted Addiction Level: **{pred_addiction:.2f} / 10**")
            st.markdown(f"### Predicted Productivity Loss: **{pred_productivity:.2f} / 10**")

            # Personalized tips box
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Personalized Tips")
            if pred_addiction > 7:
                st.warning("⚠️ Your predicted addiction level is high. Consider these tips:")
                st.write("- Set time limits on your social media apps.")
                st.write("- Switch to Wi-Fi to reduce distractions.")
                st.write("- Engage in activities like reading or exercising.")
            elif pred_addiction > 4:
                st.info("ℹ️ Moderate addiction risk. Some suggestions:")
                st.write("- Take regular breaks.")
                st.write("- Disable non-essential notifications.")
            else:
                st.success("✅ Low addiction risk. Great job!")

            if pred_productivity > 7:
                st.warning("⚠️ High productivity loss predicted. Try:")
                st.write("- Allocate specific social media time slots.")
                st.write("- Use focus apps that block distractions.")
            elif pred_productivity > 4:
                st.info("ℹ️ Moderate productivity loss. Suggestions:")
                st.write("- Prioritize important tasks.")
                st.write("- Track your screen time daily.")
            else:
                st.success("✅ Productivity loss is low. Well done!")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Interactive storytelling in sidebar
st.sidebar.header("📖 Quick Insights")
if filtered_df.empty:
    st.sidebar.write("No data matches your filters.")
else:
    avg_engagement_filtered = filtered_df["Engagement"].mean()
    avg_productivity_filtered = filtered_df["ProductivityLoss"].mean()
    common_platform = filtered_df["Platform"].mode()[0]
    st.sidebar.markdown(f"""
    - Users aged **{age_range[0]}-{age_range[1]}** on **{platform}** show average engagement of **{avg_engagement_filtered:.2f}**.
    - Average productivity loss in this group is **{avg_productivity_filtered:.2f}**.
    - Most common platform used: **{common_platform}**.
    """)

