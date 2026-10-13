import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------
# Machine Learning Imports (scikit-learn)
# ---------------------------------------------------------
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
except ModuleNotFoundError:
    st.error("⚠️ 'scikit-learn' is not installed! Please run 'pip install scikit-learn' in your terminal.")
    st.stop()

# ---------------------------------------------------------
# Page Config & Custom Dark Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Indian Airport Capacity & Delay Analytics",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background-color: #161B22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363D;
    }
    .alert-box {
        background-color: #162436;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Expanded IATA Code to City Mapping Dictionary
# ---------------------------------------------------------
AIRPORT_MAP = {
    'DEL': 'Delhi (DEL)',
    'BOM': 'Mumbai (BOM)',
    'BLR': 'Bengaluru (BLR)',
    'MAA': 'Chennai (MAA)',
    'CCU': 'Kolkata (CCU)',
    'HYD': 'Hyderabad (HYD)',
    'AMD': 'Ahmedabad (AMD)',
    'PNQ': 'Pune (PNQ)',
    'GOI': 'Goa (GOI)',
    'GOX': 'Mopa Goa (GOX)',
    'COK': 'Cochin (COK)',
    'TRV': 'Thiruvananthapuram (TRV)',
    'JAI': 'Jaipur (JAI)',
    'LKO': 'Lucknow (LKO)',
    'AGR': 'Agra (AGR)',
    'PAT': 'Patna (PAT)',
    'IXB': 'Bagdogra (IXB)',
    'GAU': 'Guwahati (GAU)',
    'VNS': 'Varanasi (VNS)',
    'BBI': 'Bhubaneswar (BBI)',
    'IXC': 'Chandigarh (IXC)',
    'NAG': 'Nagpur (NAG)',
    'IDR': 'Indore (IDR)',
    'SXR': 'Srinagar (SXR)',
    'VTZ': 'Visakhapatnam (VTZ)',
    'IXJ': 'Jammu (IXJ)',
    'ATQ': 'Amritsar (ATQ)',
    'BDQ': 'Vadodara (BDQ)',
    'TRZ': 'Tiruchirappalli (TRZ)',
    'CJB': 'Coimbatore (CJB)',
    'IXM': 'Madurai (IXM)',
    'RPR': 'Raipur (RPR)',
    'IXR': 'Ranchi (IXR)',
    'IMF': 'Imphal (IMF)',
    'DED': 'Dehradun (DED)',
    'BHO': 'Bhopal (BHO)',
    'STV': 'Surat (STV)',
    'IXA': 'Agartala (IXA)',
    'IXL': 'Leh (IXL)',
    'IXE': 'Mangalore (IXE)',
    'IXZ': 'Port Blair (IXZ)',
    'JDH': 'Jodhpur (JDH)',
    'UDR': 'Udaipur (UDR)',
    'GWL': 'Gwalior (GWL)',
    'RJA': 'Rajahmundry (RJA)',
    'DHM': 'Dharamshala (DHM)',
    'IXU': 'Aurangabad (IXU)'
}

# ---------------------------------------------------------
# Fast Unified Dataset Loader & ML Pipeline
# ---------------------------------------------------------
@st.cache_data
def load_dataset_and_model():
    file_path = "Indian-Airlines-Dataset.csv"
    try:
        raw_df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found in project folder!")
        st.stop()
        
    raw_df.columns = raw_df.columns.str.strip().str.lower()
    
    samples_per_row = 10
    df_expanded = raw_df.loc[raw_df.index.repeat(samples_per_row)].reset_index(drop=True)
    n = len(df_expanded)
    
    np.random.seed(42)
    
    start_ts = pd.Timestamp('2024-01-01').value // 10**9
    end_ts = pd.Timestamp('2024-12-31').value // 10**9
    random_timestamps = pd.to_datetime(np.random.randint(start_ts, end_ts, size=n), unit='s')
    
    df_expanded['Timestamp'] = random_timestamps
    df_expanded['Date'] = df_expanded['Timestamp'].dt.date
    df_expanded['Hour'] = df_expanded['Timestamp'].dt.hour
    df_expanded['Day'] = df_expanded['Timestamp'].dt.day_name()
    
    airlines_list = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air"]
    market_probs = [0.55, 0.20, 0.10, 0.10, 0.05]
    
    df_expanded['Airline'] = np.random.choice(airlines_list, size=n, p=market_probs)
    
    prefix_map = {'IndiGo': '6E', 'Air India': 'AI', 'SpiceJet': 'SG', 'Vistara': 'UK', 'Akasa Air': 'QP'}
    raw_num = df_expanded['fltno'].astype(str).str.extract(r'(\d+)')[0].fillna('101')
    df_expanded['Flight_Number'] = [f"{prefix_map[a]}-{num}" for a, num in zip(df_expanded['Airline'], raw_num)]

    peak_hours_mask = df_expanded['Hour'].isin([8, 9, 10, 17, 18, 19])
    is_peak = np.where(peak_hours_mask, 1.3, 0.5)
    
    utilization = np.clip(np.random.normal(50 * is_peak, 15), 5, 100)
    passengers = (utilization * 1.8).astype(int)
    tickets = (passengers * 1.02).astype(int)
    
    df_expanded['Capacity_Utilization_Ratio'] = np.round(utilization, 2)
    df_expanded['Total_Passengers_Boarded'] = passengers
    df_expanded['Total_Tickets_Sold'] = tickets
    df_expanded['Redemption_Pressure'] = np.round(np.random.uniform(0.8, 1.6, n), 2)
    
    # Logical Delay Calculation (Driven by Utilization, Peak Hours, & Redemption Pressure)
    base_delay = np.where(df_expanded['Capacity_Utilization_Ratio'] > 75, 25, 0)
    peak_delay = np.where(df_expanded['Hour'].isin([8, 9, 10, 17, 18, 19]), 15, 0)
    redemption_delay = (df_expanded['Redemption_Pressure'] - 1.0).clip(lower=0) * 20
    noise = np.random.exponential(scale=5, size=n)

    df_expanded['Departure_Delay'] = np.round(base_delay + peak_delay + redemption_delay + noise, 1)
    
    # Clean whitespace and map raw airport codes to City Name (CODE) format
    raw_source = df_expanded['source'].astype(str).str.strip().str.upper() if 'source' in df_expanded.columns else "DEL"
    raw_dest = df_expanded['dest'].astype(str).str.strip().str.upper() if 'dest' in df_expanded.columns else "BOM"
    
    df_expanded['Departure_Airport'] = raw_source.map(lambda x: AIRPORT_MAP.get(x, f"{x} Airport"))
    df_expanded['Arrival_Airport'] = raw_dest.map(lambda x: AIRPORT_MAP.get(x, f"{x} Airport"))

    month = df_expanded['Timestamp'].dt.month
    df_expanded['Season'] = np.where(month.isin([3, 4, 5]), 'Spring',
                            np.where(month.isin([6, 7, 8]), 'Summer',
                            np.where(month.isin([9, 10, 11]), 'Fall', 'Winter')))

    # --- Fast Machine Learning Training Pipeline ---
    sample_size = min(4000, len(df_expanded))
    ml_sample = df_expanded.sample(n=sample_size, random_state=42).copy()
    
    le_airline = LabelEncoder()
    le_season = LabelEncoder()
    le_airport = LabelEncoder()
    
    ml_sample['Airline_Enc'] = le_airline.fit_transform(ml_sample['Airline'])
    ml_sample['Season_Enc'] = le_season.fit_transform(ml_sample['Season'])
    ml_sample['Airport_Enc'] = le_airport.fit_transform(ml_sample['Departure_Airport'])
    
    feature_cols = ['Hour', 'Capacity_Utilization_Ratio', 'Redemption_Pressure', 'Airline_Enc', 'Season_Enc', 'Airport_Enc']
    X = ml_sample[feature_cols]
    y = ml_sample['Departure_Delay']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = RandomForestRegressor(n_estimators=35, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    encoders = {'airline': le_airline, 'season': le_season, 'airport': le_airport}
    
    return df_expanded, model, r2, mae, rmse, encoders

# Load data and ML model synchronously
df, model, r2, mae, rmse, encoders = load_dataset_and_model()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

min_d = df['Timestamp'].min().date()
max_d = df['Timestamp'].max().date()
date_range = st.sidebar.date_input("📅 Select Date Range", value=[min_d, max_d])

st.sidebar.markdown("---")

airports = sorted(df['Departure_Airport'].unique())
selected_airport = st.sidebar.selectbox("Select Departure Airport:", options=airports)

all_airlines = sorted(df['Airline'].unique())
selected_airlines = st.sidebar.multiselect("Select Airlines:", options=all_airlines, default=all_airlines)

# Filter Dataset
filtered_df = df[(df['Departure_Airport'] == selected_airport) & (df['Airline'].isin(selected_airlines))].copy()
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['Timestamp'].dt.date >= date_range[0]) & (filtered_df['Timestamp'].dt.date <= date_range[1])]

# ---------------------------------------------------------
# SECTION 1: Key Performance Indicators (KPIs)
# ---------------------------------------------------------
st.title("🛫 Airport & Airline Capacity Dashboard")

k1, k2, k3, k4 = st.columns(4)
avg_util = filtered_df['Capacity_Utilization_Ratio'].mean() if len(filtered_df) > 0 else 0
total_boarded = filtered_df['Total_Passengers_Boarded'].sum()
total_tickets = filtered_df['Total_Tickets_Sold'].sum()
avg_redemption = filtered_df['Redemption_Pressure'].mean() if len(filtered_df) > 0 else 0

k1.metric("Avg Capacity Utilization", f"{avg_util:.2f}%")
k2.metric("Total Passengers Boarded", f"{total_boarded:,}")
k3.metric("Total Tickets Sold", f"{total_tickets:,}")
k4.metric("Avg Redemption Pressure", f"{avg_redemption:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# Operational Alert
if avg_util < 20:
    alert_msg = f"💤 **Idle Capacity Alert:** Operations at **{selected_airport}** show under-utilization ({avg_util:.1f}% average)."
elif avg_util > 75:
    alert_msg = f"⚠️ **High Congestion Alert:** Terminal capacity utilization at **{selected_airport}** is high ({avg_util:.1f}% average)."
else:
    alert_msg = f"✅ **Optimal Flow:** Operations at **{selected_airport}** are running smoothly ({avg_util:.1f}% average utilization)."

st.markdown(f'<div class="alert-box">{alert_msg}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 2: Tabbed Layout
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Smooth Timeline", 
    "✈️ Airline Analysis", 
    "🗺️ Congestion Heatmap", 
    "📋 Raw Data",
    "🤖 ML Delay Predictor"
])

# TAB 1: Smooth Aggregated Timeline
with tab1:
    st.subheader("Capacity Utilization Trend")
    if len(filtered_df) > 0:
        agg_option = st.selectbox("Aggregation Level:", ["Daily Average", "Weekly Average", "Monthly Average"], index=1)
        freq_map = {"Daily Average": "D", "Weekly Average": "W", "Monthly Average": "M"}
        
        timeline_data = (
            filtered_df.set_index('Timestamp')
            .groupby([pd.Grouper(freq=freq_map[agg_option]), 'Airline'])['Capacity_Utilization_Ratio']
            .mean()
            .reset_index()
        )
        
        fig_timeline = px.line(
            timeline_data, 
            x='Timestamp', 
            y='Capacity_Utilization_Ratio',
            color='Airline',
            title=f"Capacity Utilization ({agg_option})",
            template="plotly_dark",
            labels={'Capacity_Utilization_Ratio': 'Utilization (%)', 'Timestamp': 'Date'},
            markers=True
        )
        fig_timeline.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No operational data available for the selected airport filter combination.")

# TAB 2: Airline Breakdown
with tab2:
    st.subheader("Airline Market & Performance Breakdown")
    if len(filtered_df) > 0:
        col_air1, col_air2 = st.columns(2)
        
        airline_passengers = filtered_df.groupby('Airline')['Total_Passengers_Boarded'].sum().reset_index()
        fig_share = px.pie(
            airline_passengers,
            names='Airline',
            values='Total_Passengers_Boarded',
            hole=0.4,
            title="Passenger Share by Airline",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_share.update_layout(height=400)
        col_air1.plotly_chart(fig_share, use_container_width=True)

        airline_util = filtered_df.groupby('Airline')['Capacity_Utilization_Ratio'].mean().reset_index()
        fig_util = px.bar(
            airline_util,
            x='Airline',
            y='Capacity_Utilization_Ratio',
            title="Avg Capacity Utilization (%)",
            color='Airline',
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_util.update_layout(height=400, showlegend=False)
        col_air2.plotly_chart(fig_util, use_container_width=True)
    else:
        st.info("No operational data available for the selected airport filter combination.")

# TAB 3: Heatmap & Seasonality
with tab3:
    st.subheader("Terminal Congestion & Seasonal Patterns")
    if len(filtered_df) > 0:
        col_heat, col_season = st.columns([3, 2])
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = filtered_df.groupby(['Hour', 'Day'])['Capacity_Utilization_Ratio'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='Hour', columns='Day', values='Capacity_Utilization_Ratio').reindex(index=range(24), columns=days_order).fillna(0)

        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="Day of Week", y="Hour of Day", color="Utilization (%)"),
            x=days_order,
            y=list(range(24)),
            color_continuous_scale="RdYlGn_r",
            title="Hourly Congestion Heatmap",
            template="plotly_dark"
        )
        fig_heatmap.update_layout(height=420)
        col_heat.plotly_chart(fig_heatmap, use_container_width=True)

        seasonal_df = filtered_df.groupby('Season')['Capacity_Utilization_Ratio'].mean().reset_index()
        color_map = {'Spring': '#4CAF50', 'Summer': '#F44336', 'Fall': '#FF9800', 'Winter': '#2196F3'}

        fig_season = px.bar(
            seasonal_df, 
            x='Season', 
            y='Capacity_Utilization_Ratio',
            color='Season',
            title="Seasonal Utilization",
            color_discrete_map=color_map,
            template="plotly_dark"
        )
        fig_season.update_layout(height=420, showlegend=False)
        col_season.plotly_chart(fig_season, use_container_width=True)
    else:
        st.info("No operational data available for the selected airport filter combination.")

# TAB 4: Raw Data Preview
with tab4:
    st.subheader("Data Inspector")
    if len(filtered_df) > 0:
        st.dataframe(
            filtered_df[['Timestamp', 'Airline', 'Flight_Number', 'Departure_Airport', 'Arrival_Airport', 'Capacity_Utilization_Ratio', 'Total_Passengers_Boarded', 'Departure_Delay']].head(50), 
            use_container_width=True
        )
    else:
        st.info("No operational data available for the selected airport filter combination.")

# TAB 5: Machine Learning Predictive Model
with tab5:
    st.subheader("🤖 Random Forest Departure Delay Forecasting")
    st.markdown("This tab utilizes a **Random Forest Regressor** trained on operational parameters using an **80/20 Train-Test Split (TTS)**.")
    
    # Model Evaluation Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Model R² Score", f"{r2:.3f}")
    m2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} mins")
    m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} mins")
    
    st.markdown("---")
    
    col_pred_inputs, col_pred_results = st.columns([1, 1])
    
    with col_pred_inputs:
        st.markdown("### 🎛️ Input Flight Parameters")
        input_airport = st.selectbox("Departure Airport:", options=encoders['airport'].classes_)
        input_airline = st.selectbox("Airline Carrier:", options=encoders['airline'].classes_)
        input_season = st.selectbox("Season:", options=encoders['season'].classes_)
        input_hour = st.slider("Hour of Day (0-23):", 0, 23, 9)
        input_util = st.slider("Estimated Terminal Capacity Utilization (%):", 5.0, 100.0, 75.0)
        input_redemption = st.slider("Redemption Pressure Ratio:", 0.8, 1.6, 1.1)

    with col_pred_results:
        st.markdown("### 🔮 Predicted Delay Output")
        
        airport_e = encoders['airport'].transform([input_airport])[0]
        airline_e = encoders['airline'].transform([input_airline])[0]
        season_e = encoders['season'].transform([input_season])[0]
        
        input_vector = np.array([[input_hour, input_util, input_redemption, airline_e, season_e, airport_e]])
        
        predicted_delay = model.predict(input_vector)[0]
        
        st.markdown(f"<h1 style='color:#1E88E5; font-size:48px;'>{max(0, round(predicted_delay, 1))} Mins</h1>", unsafe_allow_html=True)
        
        if predicted_delay < 15:
            st.success("🟢 **Low Delay Risk:** Operational flow is expected to be smooth.")
        elif predicted_delay < 35:
            st.warning("🟡 **Moderate Delay Risk:** Minor terminal congestion expected.")
        else:
            st.error("🔴 **High Delay Risk:** Severe gate bottleneck and departure delays anticipated.")
            
        st.markdown("#### Feature Importance Rankings")
        importance_df = pd.DataFrame({
            'Feature': ['Hour', 'Utilization', 'Redemption', 'Airline', 'Season', 'Airport'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', template="plotly_dark", height=220)
        fig_imp.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)